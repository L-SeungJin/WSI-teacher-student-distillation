# WSI-teacher-student-distillation
Teacher–student distillation framework for efficient tissue microenvironment classification in whole slide images (WSIs).

---

## 1. Overview 
Whole Slide Images (WSIs) contain millions of pixels and thousands of tissue patches, making direct end-to-end training computationally expensive.

This project implements a **teacher–student distillation pipeline** where:

- A **teacher model** generates patch-level predictions
- A **student model** learns from those predictions
- Spatial relationships between patches are modeled using **local band attention and similarity constraints**

The goal is to build an efficient model for **tissue microenvironment classification in WSIs**.

---

## 2. Motivation 
Processing WSIs directly with deep learning models is challenging due to:

- Extremely large image sizes
- High computational cost
- Limited labeled data

Teacher–student distillation helps address these issues by:

- Leveraging teacher predictions as supervision
- Training a lightweight student model
- Modeling spatial context between neighboring patches

We further analyze how architectural components affect performance through **ablation experiments**.

---

## 3. Method 
The student model processes patch features extracted from WSIs and learns spatial relationships between patches.

Main components:

**Input**

- Patch features extracted from a pretrained encoder
- Patch spatial coordinates

**Student architecture**

- Linear projection of patch features
- 2D positional encoding
- Local band attention for spatial context
- Similarity-based attention bias (CAMIL-style)
- Global context aggregation

**Prediction**

- Patch-level tissue classification
- Visualization through WSI overlay

---

## 4. Repository Structure
```
WSI-teacher-student-distillation
│
├ code
│   ├ teacher
│   ├ student
│   ├ inference
│   └ analysis
│
├ configs
├ environment
├ experiments
├ pipeline
├ results
│   ├ figures
│   └ tables
│
├ requirements.txt
└ README.md
```
---

## 5. Dataset 
This project uses gastric cancer Whole Slide Images (WSIs) obtained through a clinical collaboration with **Seoul National University Hospital (SNUH)**.

Dataset characteristics:

- Institution: Seoul National University Hospital
- Data type: Gastric cancer Whole Slide Images (WSIs)
- Availability: Not publicly available due to clinical data privacy restrictions

WSIs are divided into patches, and patch-level features are extracted using a pretrained encoder.

Example tissue classes:

- ADI (Adipose)
- STR (Stroma)
- LYM (Lymphocyte)
- MUC (Mucus)
- MUS (Muscle)
- NOR (Normal)
- TUM (Tumor)
- DEB (Debris)

Patch coordinates are used to model spatial relationships between tissue regions.
---

## 6. Environment 
Experiments were conducted in a Docker-based environment.

Install dependencies:
```
pip install -r requirements.txt
```

Main dependencies:

Python 3.11  
PyTorch 2.10  
CUDA 12.8  

Key libraries:

- tiatoolbox
- torchvision
- timm
- opencv-python
- numpy
- scikit-learn
- segmentation_models_pytorch

Full environment packages:

See `environment/pip_list.txt`

---

## 7. Reproducibility Note

This repository focuses on the model architecture, training pipeline, and experiment setup.

The original WSI dataset used in this study was obtained through a clinical collaboration with Seoul National University Hospital and cannot be publicly released due to data privacy restrictions.

Therefore, the exact reproduction of the reported results is not possible without access to the original dataset. However, the codebase can be adapted to other WSI datasets for similar experiments.

---

## 8. Training Pipeline 

1. Generate teacher predictions
Teacher predictions are assumed to be precomputed from a separate teacher model.
```bash
python code/teacher/make_teacher_from_csv.py
```
2. Train student model
```bash
python code/student/train_student_band_camil_longmil.py
```
3. Run inference
```bash
python code/inference/infer_student_band_camil_longmil.py
```
---

## 9. Experiments 
We conduct ablation experiments to analyze the impact of different architectural components.

Compared variants:

- **Full model** (Band + CAMIL similarity + Global context)
- **No Global** (without global aggregation)
- **No Similarity** (without similarity bias)

Each model is evaluated using **Macro-F1 score**.

---

## 10. Results 
### Ablation Results

Macro-F1 scores for different student model variants.

| Model | Macro-F1 |
|------|---------|
| Full (Band + CAMIL + Global) | 0.8593 |
| No Global | 0.8599 |
| No Similarity | 0.8601 |

### Qualitative Comparison
Example visualization of student predictions on a representative WSI.
![Ablation Panel](results/figures/AGT-001-6_ablation_panel.png)

---

## 11. Future Work

Possible future improvements include:

- Incorporating multi-scale WSI features
- Improving spatial context modeling
- Applying the framework to other cancer types
- Integrating graph neural networks for spatial tissue modeling

---

## 12. Discussion and Limitations

In this study, experiments were conducted primarily at the **patch-level classification setting**.

We adopted a patch-level formulation because WSIs are extremely large, and patch-level features allow efficient modeling of **local spatial relationships between tissue regions**. This setting also makes it easier to analyze the effect of architectural components such as band attention, similarity bias, and global context aggregation.

However, the ultimate clinical task in computational pathology is often **slide-level prediction** (e.g., diagnosis or survival prediction). In our experiments, the full model does not consistently outperform the ablation variants, which may indicate that the benefits of spatial modeling are not fully captured at the patch-level classification setting.

Future work will extend this framework to **slide-level learning settings**, such as multiple instance learning (MIL) or graph-based WSI modeling, where spatial relationships between patches may play a more significant role.
