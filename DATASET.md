# 📦 Dataset Overview

This document introduces the TMCAD dataset used in this project. TMCAD stands for **truly mechanical CAD dataset**.

- The TMCAD dataset v2, 2025.11.02
- The TMCAD dataset v1, 2025.02.10

---

## 📘 Dataset Description

TMCAD v2 contains **9,799 CAD models** spanning **10 standardized mechanical part categories**.  
All models are provided in **STEP (.stp)** format with standardized naming conventions.
It was collected as part of the BRT to support research in learning from CAD models. 

### Dataset Overview

| Category   | # of Models |
| ---------- | ----------: |
| Bearing    |         857 |
| Bolt-Screw |       2,070 |
| Bracket    |       1,102 |
| Flange     |         979 |
| Gear       |         962 |
| Nut        |         899 |
| Shaft      |         893 |
| Coupling   |         474 |
| Pulley     |         544 |
| Spring     |       1,019 |
| **Total**  |   **9,799** |


---

## Download and Use

The dataset is publicly available at the following location:

### TMCAD v2
The TMCAD dataset v2 (a refined version released on 2025.11.02) can be downloaded from: https://pan.zju.edu.cn/share/218d10a88e8c18f5b96e94a7e0


**TMCAD v2**  is a **cleaned and verified collection of 3D CAD mechanical parts**, designed for research in 3D **shape classification**, **representation learning**, and **retrieval**, etc. The specific improvements in v2 are:
- Removed mistakenly collected samples  
- Corrected wrong or ambiguous class labels  
- Verified and repaired invalid geometries  
- Merged visually and functionally similar classes (`bolt` + `screw`)  
- Ensured clean hierarchical directory structure


### TMCAD v1
The TMCAD dataset v1 (2025.02.10) can be downloaded from: https://pan.zju.edu.cn/share/305e1697a37277e6a9ec60dded

### Contents

TMCAD_v2/
 │
 ├── bearing/
 │   ├── bearing_0.stp
 │   ├── bearing_1.stp
 │   └── ...
 │
 ├── bolt-screw/
 │   ├── bolt_0.stp
 │   ├── bolt_1.stp
 │   └── ...
 │
 ├── ...
 │
 └── spring/
 ├── spring_0.stp
 ├── spring_1.stp
 └── ...

Each folder corresponds to a **part category** and contains 3D CAD files in `.stp` format.


---

## Applications

TMCAD v2 can be directly used for a wide range of 3D geometric learning tasks:

- 🧭 **3D Classification** — Train networks like PointNet, DGCNN, Point Transformer  
- 🔍 **Shape Retrieval** — Evaluate similarity-based methods on mechanical components  
- 🧩 **Few-shot / Meta-learning** — Build n-way k-shot episodic datasets  
- 🧠 **Representation Learning** — Pre-train or fine-tune on CAD-level geometry  
- ⚙️ **Cross-modal Tasks** — Combine with textual or image data for multi-modal learning  


## License
The dataset is released under GPL-3.0 license.

## Citation
If you use this dataset, please cite:
@article{zou2025bringing,
  title={Bringing attention to CAD: Boundary representation learning via transformer},
  author={Zou, Qiang and Zhu, Lizhen},
  journal={Computer-Aided Design},
  pages={103940},
  year={2025},
  publisher={Elsevier}
}