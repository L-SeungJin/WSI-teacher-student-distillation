import os
from glob import glob
from pathlib import Path
from PIL import Image, ImageDraw

ROOT = "/workspace/GC_WSI_Results_MoE_UNI"
OUT_ROOT = "/workspace/my_exp/compare_teacher_student_all"
os.makedirs(OUT_ROOT, exist_ok=True)

def load_rgb(path):
    return Image.open(path).convert("RGB")

def fit_height(img, target_h):
    w, h = img.size
    scale = target_h / h
    return img.resize((int(w * scale), target_h), Image.BILINEAR)

slide_dirs = sorted(glob(os.path.join(ROOT, "AGT-*")))
print("found slide dirs:", len(slide_dirs))

saved = 0
missing = 0

for slide_dir in slide_dirs:
    sid = Path(slide_dir).name

    teacher_path = Path(slide_dir) / f"{sid}_overlay_moe_uni.png"
    student_path = Path(slide_dir) / f"{sid}_overlay_student_band_camil_longmil.png"

    if not teacher_path.exists():
        print(f"[MISS teacher] {sid}")
        missing += 1
        continue
    if not student_path.exists():
        print(f"[MISS student] {sid}")
        missing += 1
        continue

    teacher = load_rgb(teacher_path)
    student = load_rgb(student_path)

    target_h = min(teacher.height, student.height, 1400)
    teacher = fit_height(teacher, target_h)
    student = fit_height(student, target_h)

    margin = 30
    title_h = 80

    W = teacher.width + student.width + margin * 3
    H = target_h + title_h + margin * 2

    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    draw.text((margin, 15), f"{sid}", fill=(0, 0, 0))
    draw.text((margin, 42), "Left: Teacher (MoE-UNI)   |   Right: Student (Band + CAMIL + LongMIL)", fill=(0, 0, 0))

    y0 = title_h
    canvas.paste(teacher, (margin, y0))
    canvas.paste(student, (teacher.width + margin * 2, y0))

    draw.text((margin, y0 - 24), "Teacher", fill=(0, 0, 0))
    draw.text((teacher.width + margin * 2, y0 - 24), "Student", fill=(0, 0, 0))

    out_path = Path(OUT_ROOT) / f"{sid}_compare_teacher_student.png"
    canvas.save(out_path)
    print("saved:", out_path)
    saved += 1

print(f"done. saved={saved}, missing={missing}")
