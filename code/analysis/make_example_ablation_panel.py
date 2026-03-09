import cv2
import numpy as np

BASE = "/workspace/GC_WSI_Results_MoE_UNI/AGT-001-6"

img_full = cv2.imread(BASE + "/AGT-001-6_overlay_student_band_camil_longmil.png")
img_noglobal = cv2.imread(BASE + "/AGT-001-6_overlay_student_band_noglobal.png")
img_nosim = cv2.imread(BASE + "/AGT-001-6_overlay_student_band_nosim.png")

# 크기 맞추기
h = min(img_full.shape[0], img_noglobal.shape[0], img_nosim.shape[0])

img_full = cv2.resize(img_full, (int(img_full.shape[1]*h/img_full.shape[0]), h))
img_noglobal = cv2.resize(img_noglobal, (int(img_noglobal.shape[1]*h/img_noglobal.shape[0]), h))
img_nosim = cv2.resize(img_nosim, (int(img_nosim.shape[1]*h/img_nosim.shape[0]), h))

panel = np.hstack([img_full, img_noglobal, img_nosim])

cv2.imwrite("results/figures/AGT-001-6_ablation_panel.png", panel)

print("saved panel")