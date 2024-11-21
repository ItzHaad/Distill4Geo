# Distill4Geo: Efficient Cross-View Geo-Localization with Lightweight Models

Welcome to the official repository of **Distill4Geo**, a novel knowledge distillation framework designed for cross-view geo-localization. This repository accompanies the paper *Distill4Geo: Streamlined Knowledge Transfer from Contrastive Weight-Sharing Teachers to Independent, Lightweight View Experts*, **currently under review at CVPR 2025**.

---

## 📖 Overview

Cross-View Geo-Localization (CVGL) aims to match images from different perspectives—like satellite and street views—to the same geographic location. Distill4Geo addresses the computational overhead and inefficiencies in traditional contrastive methods by proposing:

- **Lightweight, non-weight-sharing student models** for independent view representation.
- **Dual Cosine Embedding Loss** for efficient knowledge transfer from a contrastive-trained teacher model.
- **Data Augmentation with Marginal Perturbations** to improve pair-wise generalization.

Our approach achieves **state-of-the-art performance** with significant reductions in parameters (3×) and GFLOPs (13.5×) compared to existing models.

---

## 🚀 Features

- **Efficient Knowledge Transfer**: A teacher-student framework eliminates the need for large batch sizes and hard-negative mining.
- **Dual Distillation**: Uses cosine embedding loss to bring each student closer to the teacher's representations.
- **Lightweight Students**: FasterViT-based architecture for resource-efficient training and inference.
- **Generalization Across Datasets**: Achieves competitive results on CVUSA, CVACT, and VIGOR datasets.

---

## 🏆 Highlights

- **Parameters**: Reduced 3× compared to the teacher model.
- **GFLOPs**: Over 13.5× lower computational cost.
- **Performance**: Competitive accuracy across datasets.
- **Scalability**: Independent student training enables resource-efficient deployment.

---

## 📊 Results

### CVUSA Datasets
| Method         | R@1    | R@5    | R@10   | R@1%   |
|----------------|---------|---------|---------|---------|
| Pass-KD        | 94.09% | 98.42% | 99.13% | 99.77% |
| Sample4Geo     | 98.68% | 99.68% | 99.78% | 99.87% |
| **Distill4Geo** | 97.92% | 99.58% | 99.76% | 99.86% |

### CVACT Test Datasets
| Method         | R@1    | R@5    | R@10   | R@1%   |
|----------------|---------|---------|---------|---------|
| Pass-KD        | 66.81% | 88.03% | 90.87% | 98.02% |
| Sample4Geo     | 71.51% | 92.42% | 94.45% | 97.90% |
| **Distill4Geo** | 71.07% | 91.66% | 93.83% | 98.76% |

### VIGOR Dataset (Cross-Split)
| Method         | R@1    | R@5    | R@10   | Hit Rate |
|----------------|---------|---------|---------|----------|
| Sample4Geo     | 61.70% | 83.50% | 88.00% | 69.87%   |
| **Distill4Geo** | 59.93% | 83.09% | 87.66% | 68.14%   |

---

## 📂 Repository Structure

- `src/` - Core implementation of Distill4Geo, including teacher-student training modules.
- `datasets/` - Scripts for preparing and loading CVUSA, CVACT, and VIGOR datasets.
- `notebooks/` - Jupyter notebooks for result visualization and evaluation.
- `experiments/` - Configurations and logs for reproducibility.

---

## 📜 Citation

If you find our work useful, please cite:

```bibtex
@article{distill4geo2025,
  title={Distill4Geo: Streamlined Knowledge Transfer from Contrastive Weight-Sharing Teachers to Independent, Lightweight View Experts},
  author={Anonymous},
  journal={CVPR},
  year={2025}
}

## 🙌 Acknowledgements

We would like to extend our gratitude to the authors of the following projects, which greatly inspired and influenced this work:

- **FasterViT**: A fast hybrid vision transformer architecture that combines the efficiency of CNNs with the global modeling capabilities of Vision Transformers. Special thanks to the authors for their contributions to efficient deep learning.  
  Repository: [FasterViT GitHub](https://github.com/NVlabs/FasterViT)

- **Sample4Geo**: A state-of-the-art framework for cross-view geo-localization using contrastive learning and hard negative mining. We appreciate the authors for providing an excellent foundation for our knowledge distillation work.  
  Repository: [Sample4Geo GitHub](https://github.com/fdeuser/Sample4Geo)

Your innovative work has been instrumental in advancing the field of cross-view geo-localization!

---

## 🔗 References

This repository builds upon prior works in cross-view geo-localization and knowledge distillation. See the [paper](link) for detailed references.

---

Feel free to star ⭐ this repository and follow for updates!
