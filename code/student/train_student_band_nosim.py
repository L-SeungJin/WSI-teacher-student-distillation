import os, glob, math, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

DATA_ROOT = "/workspace/GC_WSI_Results_MoE_UNI"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_CLASSES = 8
FEAT_DIM = 1024

MAX_PATCHES = 4096
BATCH_SIZE = 1
EPOCHS = 20
LR = 1e-4
SEED = 42

WINDOW_RADIUS = 2
CE_WEIGHT = 0.3
SIM_BETA = 0.0   # 핵심: similarity 제거

SAVE_DIR = "/workspace/my_exp/student_ckpt_band_nosim"
os.makedirs(SAVE_DIR, exist_ok=True)

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_everything(SEED)

def get_slide_dirs(root):
    dirs = []
    for d in sorted(glob.glob(os.path.join(root, "AGT-*"))):
        sid = os.path.basename(d)
        feat_p = os.path.join(d, f"{sid}_feats_uni.npy")
        coord_p = os.path.join(d, f"{sid}_coords.npy")
        prob_p = os.path.join(d, f"{sid}_probs.npy")
        pred_p = os.path.join(d, f"{sid}_preds.npy")
        if all(os.path.exists(p) for p in [feat_p, coord_p, prob_p, pred_p]):
            dirs.append(d)
    return dirs

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

def macro_f1_score(y_true, y_pred, num_classes=8):
    f1s = []
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    for c in range(num_classes):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        p = tp / (tp + fp + 1e-8)
        r = tp / (tp + fn + 1e-8)
        f1 = 2 * p * r / (p + r + 1e-8)
        f1s.append(f1)
    return float(np.mean(f1s))

class SlidePatchDataset(Dataset):
    def __init__(self, slide_dirs, max_patches=4096, train=True):
        self.slide_dirs = slide_dirs
        self.max_patches = max_patches
        self.train = train

    def __len__(self):
        return len(self.slide_dirs)

    def __getitem__(self, idx):
        d = self.slide_dirs[idx]
        sid = os.path.basename(d)

        feats = np.load(os.path.join(d, f"{sid}_feats_uni.npy"))
        coords = np.load(os.path.join(d, f"{sid}_coords.npy"))
        probs = np.load(os.path.join(d, f"{sid}_probs.npy"))
        preds = np.load(os.path.join(d, f"{sid}_preds.npy"))

        n = len(feats)
        if n > self.max_patches:
            if self.train:
                sel = np.random.choice(n, self.max_patches, replace=False)
            else:
                sel = np.arange(self.max_patches)
            feats = feats[sel]
            coords = coords[sel]
            probs = probs[sel]
            preds = preds[sel]

        coords_raw = coords.astype(np.int32)
        coords_norm = coords.astype(np.float32)
        coords_norm[:, 0] /= (coords_norm[:, 0].max() + 1e-6)
        coords_norm[:, 1] /= (coords_norm[:, 1].max() + 1e-6)

        stride = infer_patch_stride(coords_raw)
        grid = np.round(coords_raw / float(stride)).astype(np.int32)

        return {
            "sid": sid,
            "feats": torch.tensor(feats, dtype=torch.float32),
            "coords_norm": torch.tensor(coords_norm, dtype=torch.float32),
            "grid": torch.tensor(grid, dtype=torch.int32),
            "probs": torch.tensor(probs, dtype=torch.float32),
            "preds": torch.tensor(preds, dtype=torch.long),
        }

def collate_fn(batch):
    return batch[0]

all_slide_dirs = get_slide_dirs(DATA_ROOT)
print("slides found:", len(all_slide_dirs))
train_dirs = all_slide_dirs[:-5]
val_dirs = all_slide_dirs[-5:]
print("train:", len(train_dirs), "val:", len(val_dirs))

train_ds = SlidePatchDataset(train_dirs, max_patches=MAX_PATCHES, train=True)
val_ds = SlidePatchDataset(val_dirs, max_patches=MAX_PATCHES, train=False)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

class Pos2DMLP(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
    def forward(self, coords):
        return self.net(coords)

class LocalBandSimilarityBlock(nn.Module):
    def __init__(self, d_model, radius=2, sim_beta=0.0, dropout=0.1):
        super().__init__()
        self.radius = radius
        self.sim_beta = sim_beta
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
            nn.Linear(d_model * 4, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x, grid):
        h = self.norm1(x)
        N, D = h.shape
        q = self.q_proj(h)
        k = self.k_proj(h)
        v = self.v_proj(h)
        out = torch.zeros_like(h)
        grid_f = grid.float()

        for i in range(N):
            gx = grid_f[i, 0]
            gy = grid_f[i, 1]
            dx = torch.abs(grid_f[:, 0] - gx)
            dy = torch.abs(grid_f[:, 1] - gy)
            nbr = torch.where((dx <= self.radius) & (dy <= self.radius))[0]
            nbr = nbr[nbr != i]

            if len(nbr) == 0:
                out[i] = v[i]
                continue

            qi = q[i:i+1]
            kj = k[nbr]
            vj = v[nbr]

            attn = (qi @ kj.T) / math.sqrt(D)
            # similarity 제거: sim_beta=0
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

class StudentNet(nn.Module):
    def __init__(self, feat_dim=1024, d_model=512, num_classes=8, radius=2, sim_beta=0.0):
        super().__init__()
        self.in_proj = nn.Linear(feat_dim, d_model)
        self.pos2d = Pos2DMLP(d_model)
        self.local1 = LocalBandSimilarityBlock(d_model, radius=radius, sim_beta=sim_beta)
        self.local2 = LocalBandSimilarityBlock(d_model, radius=radius, sim_beta=sim_beta)
        self.global_block = GlobalContextBlock(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, feats, coords_norm, grid):
        x = self.in_proj(feats) + self.pos2d(coords_norm)
        x = self.local1(x, grid)
        x = self.local2(x, grid)
        x = self.global_block(x)
        logits = self.head(x)
        return logits

def distill_loss(student_logits, teacher_probs, teacher_preds, ce_weight=0.3):
    log_p = F.log_softmax(student_logits, dim=-1)
    kl = F.kl_div(log_p, teacher_probs, reduction="batchmean")
    ce = F.cross_entropy(student_logits, teacher_preds)
    return kl + ce_weight * ce, kl.item(), ce.item()

model = StudentNet(
    feat_dim=FEAT_DIM,
    d_model=512,
    num_classes=NUM_CLASSES,
    radius=WINDOW_RADIUS,
    sim_beta=SIM_BETA,
).to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

def run_epoch(loader, train=True):
    model.train() if train else model.eval()
    total_loss = total_kl = total_ce = 0.0
    all_true, all_pred = [], []

    for batch in loader:
        feats = batch["feats"].to(DEVICE)
        coords_norm = batch["coords_norm"].to(DEVICE)
        grid = batch["grid"].to(DEVICE)
        t_probs = batch["probs"].to(DEVICE)
        t_preds = batch["preds"].to(DEVICE)

        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            s_logits = model(feats, coords_norm, grid)
            loss, kl_v, ce_v = distill_loss(s_logits, t_probs, t_preds, ce_weight=CE_WEIGHT)
            if train:
                loss.backward()
                optimizer.step()

        total_loss += loss.item()
        total_kl += kl_v
        total_ce += ce_v

        pred = s_logits.argmax(dim=1).detach().cpu().numpy()
        true = t_preds.detach().cpu().numpy()
        all_pred.extend(pred.tolist())
        all_true.extend(true.tolist())

    n = len(loader)
    return {
        "loss": total_loss / n,
        "kl": total_kl / n,
        "ce": total_ce / n,
        "macro_f1": macro_f1_score(all_true, all_pred, num_classes=NUM_CLASSES),
    }

best_f1 = -1.0

for epoch in range(1, EPOCHS + 1):
    tr = run_epoch(train_loader, train=True)
    va = run_epoch(val_loader, train=False)

    print(
        f"[Epoch {epoch:02d}] "
        f"train loss={tr['loss']:.4f} kl={tr['kl']:.4f} ce={tr['ce']:.4f} f1={tr['macro_f1']:.4f} | "
        f"val loss={va['loss']:.4f} kl={va['kl']:.4f} ce={va['ce']:.4f} f1={va['macro_f1']:.4f}"
    )

    if va["macro_f1"] > best_f1:
        best_f1 = va["macro_f1"]
        save_path = os.path.join(SAVE_DIR, "best_student.pt")
        torch.save({
            "model": model.state_dict(),
            "best_val_f1": best_f1,
            "config": {
                "feat_dim": FEAT_DIM,
                "num_classes": NUM_CLASSES,
                "window_radius": WINDOW_RADIUS,
                "sim_beta": SIM_BETA,
                "max_patches": MAX_PATCHES,
                "use_global": True,
            }
        }, save_path)
        print(f"  -> saved best to {save_path}")

print("done. best val macro-F1 =", best_f1)
