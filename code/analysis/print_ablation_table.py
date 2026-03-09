import torch
import os

experiments = {
    "Full (Band + CAMIL + Global)" :
        "/workspace/my_exp/student_ckpt_band_camil_longmil/best_student.pt",

    "No Global" :
        "/workspace/my_exp/student_ckpt_band_noglobal/best_student.pt",

    "No Similarity" :
        "/workspace/my_exp/student_ckpt_band_nosim/best_student.pt",
}

results = []

for name, path in experiments.items():

    if not os.path.exists(path):
        results.append((name, "NOT FOUND"))
        continue

    ckpt = torch.load(path, map_location="cpu")

    if "best_val_f1" in ckpt:
        f1 = ckpt["best_val_f1"]
    else:
        f1 = "UNKNOWN"

    results.append((name, f1))


print("\n===== Ablation Results =====\n")

print("{:<35} {:>10}".format("Model", "Macro-F1"))
print("-"*50)

for name, f1 in results:

    if isinstance(f1, float):
        print("{:<35} {:>10.4f}".format(name, f1))
    else:
        print("{:<35} {:>10}".format(name, f1))

print("\n")
