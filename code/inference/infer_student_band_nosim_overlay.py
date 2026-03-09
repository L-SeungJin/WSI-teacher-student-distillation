import os, glob, math
import numpy as np
import cv2
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

from tiatoolbox.wsicore.wsireader import WSIReader


DEVICE = "cpu"

WSI_ROOT = "/data2/RAW_DATA/GC_WSI_SNU/AGC_HE-30"
DATA_ROOT = "/workspace/GC_WSI_Results_MoE_UNI"
CKPT_PATH = "/workspace/my_exp/student_ckpt_band_nosim/best_student.pt"

CLASS_NAMES = ["ADI", "DEB", "LYM", "MUC", "MUS", "NOR", "STR", "TUM"]

COLOR_PALETTE = {
    "ADI": (255, 200, 60),
    "STR": (220, 220, 220),
    "DEB": (100, 180, 250),
    "LYM": (60, 160, 60),
    "MUC": (240, 180, 240),
    "MUS": (200, 90, 90),
    "NOR": (180, 180, 90),
    "TUM": (60, 60, 220),
}

COLOR_NAME_EN = {
    "STR": "Gray",
    "ADI": "Orange",
    "DEB": "Sky Blue",
    "LYM": "Green",
    "MUC": "Lavender",
    "MUS": "Red",
    "NOR": "Light Khaki",
    "TUM": "Blue",
}


def infer_patch_stride(coords):
    xs = np.unique(coords[:, 0])
    ys = np.unique(coords[:, 1])

    dx = np.diff(np.sort(xs))
    dy = np.diff(np.sort(ys))

    dx = dx[dx > 0]
    dy = dy[dy > 0]

    cand = []
    if len(dx) > 0:
        cand.append(np.min(dx))
    if len(dy) > 0:
        cand.append(np.min(dy))

    if len(cand) == 0:
        return 256

    return int(min(cand))


def make_thumbnail(wsi, max_size=2000):
    W0, H0 = wsi.info.level_dimensions[0]

    level = len(wsi.info.level_dimensions) - 1
    low_w, low_h = wsi.info.level_dimensions[level]

    try:
        arr = wsi.read_rect(
            location=(0, 0),
            size=(low_w, low_h),
            resolution=level,
            units="level",
        )
    except TypeError:
        arr = wsi.read_rect((0, 0), (low_w, low_h), level, "level")

    if isinstance(arr, Image.Image):
        arr = np.array(arr)

    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]

    h, w = arr.shape[:2]
    resize_factor = min(max_size / max(h, w), 1.0)
    new_w = int(w * resize_factor)
    new_h = int(h * resize_factor)

    arr = cv2.resize(arr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    scale_x = new_w / W0
    scale_y = new_h / H0
    scale = min(scale_x, scale_y)

    return Image.fromarray(arr), scale


def draw_overlay(thumb_pil, scale, coords, patch_size, pred_idx):
    base = np.array(thumb_pil.convert("RGB"))
    canvas = base.copy()

    for (x, y), c in zip(coords, pred_idx):
        cname = CLASS_NAMES[int(c)]
        color = COLOR_PALETTE[cname]

        x0, y0 = int(x * scale), int(y * scale)
        x1, y1 = int((x + patch_size) * scale), int((y + patch_size) * scale)

        cv2.rectangle(canvas, (x0, y0), (x1, y1), color, -1)

    over = cv2.addWeighted(base, 0.4, canvas, 0.6, 0)

    legend_x = 30
    legend_y = 40

    for cname in CLASS_NAMES:
        color = COLOR_PALETTE[cname]
        cv2.rectangle(over, (legend_x, legend_y - 18), (legend_x + 40, legend_y + 8), color, -1)

        text = f"{cname}: {COLOR_NAME_EN[cname]}"
        cv2.putText(
            over,
            text,
            (legend_x + 50, legend_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            1,
            cv2.LINE_AA
        )
        legend_y += 32

    return over


class Pos2DMLP(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )

    def forward(self, coords):
        return self.net(coords)


class LocalBandNoSimBlock(nn.Module):
    def __init__(self, d_model, radius=2, dropout=0.1):
        super().__init__()
        self.radius = radius

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )

        self.drop = nn.Dropout(dropout)

    def forward(self, x, grid):
        h = self.norm1(x)
        N, D = h.shape

        q = self.q_proj(h)
        k = self.k_proj(h)
        v = self.v_proj(h)

        out = torch.zeros_like(h)

        for i in range(N):
            gx = grid[i, 0]
            gy = grid[i, 1]

            dx = torch.abs(grid[:, 0] - gx)
            dy = torch.abs(grid[:, 1] - gy)

            nbr = torch.where((dx <= self.radius) & (dy <= self.radius))[0]
            nbr = nbr[nbr != i]

            if len(nbr) == 0:
                out[i] = v[i]
                continue

            qi = q[i:i+1]
            kj = k[nbr]
            vj = v[nbr]

            attn = (qi @ kj.T) / math.sqrt(D)
            attn = F.softmax(attn, dim=-1)

            out[i] = attn @ vj

        x = x + self.drop(self.o_proj(out))
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x


class GlobalContextBlock(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()

        self.cls_token = nn.Parameter(torch.zeros(1, d_model))

        self.norm = nn.LayerNorm(d_model)

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        self.out_proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.norm(x)
        cls = self.cls_token.expand(1, -1)

        q = self.q_proj(cls)
        k = self.k_proj(h)
        v = self.v_proj(h)

        attn = (q @ k.T) / math.sqrt(h.size(1))
        attn = F.softmax(attn, dim=-1)

        g = attn @ v
        g = self.out_proj(g)

        x = x + self.drop(g.expand_as(x))
        return x


class StudentNetNoSim(nn.Module):
    def __init__(self):
        super().__init__()
        d_model = 512

        self.in_proj = nn.Linear(1024, d_model)
        self.pos2d = Pos2DMLP(d_model)

        self.local1 = LocalBandNoSimBlock(d_model, radius=2)
        self.local2 = LocalBandNoSimBlock(d_model, radius=2)

        self.global_block = GlobalContextBlock(d_model)
        self.head = nn.Linear(d_model, 8)

    def forward(self, feats, coords_norm, grid):
        x = self.in_proj(feats) + self.pos2d(coords_norm)
        x = self.local1(x, grid)
        x = self.local2(x, grid)
        x = self.global_block(x)
        logits = self.head(x)
        return logits


print("DEVICE =", DEVICE)

model = StudentNetNoSim().to(DEVICE)
ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
model.load_state_dict(ckpt["model"])
model.eval()

slide_dirs = sorted(glob.glob(DATA_ROOT + "/AGT-*"))
print("slides:", len(slide_dirs))

for d in slide_dirs:
    sid = os.path.basename(d)
    print("processing:", sid)

    feat_path = f"{d}/{sid}_feats_uni.npy"
    coord_path = f"{d}/{sid}_coords.npy"

    if not os.path.exists(feat_path) or not os.path.exists(coord_path):
        print("missing input npy:", sid)
        continue

    feats = np.load(feat_path)
    coords = np.load(coord_path)

    stride = infer_patch_stride(coords)
    grid = np.round(coords / float(stride)).astype(np.int32)

    coords_norm = coords.astype(np.float32)
    coords_norm[:, 0] /= (coords_norm[:, 0].max() + 1e-6)
    coords_norm[:, 1] /= (coords_norm[:, 1].max() + 1e-6)

    feats_t = torch.tensor(feats, dtype=torch.float32, device=DEVICE)
    coords_norm_t = torch.tensor(coords_norm, dtype=torch.float32, device=DEVICE)
    grid_t = torch.tensor(grid, dtype=torch.float32, device=DEVICE)

    with torch.no_grad():
        logits = model(feats_t, coords_norm_t, grid_t)
        preds = logits.argmax(1).cpu().numpy()

    svs_list = glob.glob(f"{WSI_ROOT}/{sid}.*")
    if len(svs_list) == 0:
        print("missing svs:", sid)
        continue

    svs = svs_list[0]
    wsi = WSIReader.open(svs)
    thumb, scale = make_thumbnail(wsi)

    try:
        mpp_x, mpp_y = wsi.info.mpp
        current_mpp = float(np.mean([mpp_x, mpp_y]))
    except Exception:
        current_mpp = 0.5

    TARGET_MPP = 0.5
    scale_factor = TARGET_MPP / current_mpp
    BASE_PATCH_SIZE = 256
    patch_size = int(BASE_PATCH_SIZE * scale_factor)

    overlay = draw_overlay(thumb, scale, coords, patch_size, preds)

    out_path = f"{d}/{sid}_overlay_student_band_nosim.png"
    Image.fromarray(overlay).save(out_path)

    print("saved:", out_path)

print("done")