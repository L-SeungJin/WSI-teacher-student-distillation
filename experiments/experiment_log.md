# Experiment Log

This document records experiments conducted for the WSI teacher–student distillation project.

---

## Experiment 1: Full Model

Model configuration:
- Band Attention
- CAMIL Similarity Bias
- Global Context Block

Training script:

python code/student/train_student_band_camil_longmil.py

Checkpoint:

/workspace/my_exp/student_ckpt_band_camil_longmil/best_student.pt

Result:

Macro-F1 = **0.8593**

Notes:
- Baseline student architecture
- Includes both local spatial context and global aggregation

---

## Experiment 2: No Global Context

Training script:

python code/student/train_student_band_noglobal.py

Result:

Macro-F1 = **0.8599**

Notes:
- Removing global context slightly improved performance

---

## Experiment 3: No Similarity Bias

Training script:

python code/student/train_student_band_nosim.py

Result:

Macro-F1 = **0.8601**

Notes:
- Removing similarity constraint slightly improved performance