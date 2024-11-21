Distill4Geo: Efficient Cross-View Geo-Localization with Lightweight Models
Welcome to the official repository of Distill4Geo, a novel knowledge distillation framework designed for cross-view geo-localization. This repository accompanies the paper Distill4Geo: Streamlined Knowledge Transfer from Contrastive Weight-Sharing Teachers to Independent, Lightweight View Experts, currently under review at CVPR 2025.

📖 Overview
Cross-View Geo-Localization (CVGL) aims to match images from different perspectives—like satellite and street views—to the same geographic location. Distill4Geo addresses the computational overhead and inefficiencies in traditional contrastive methods by proposing:

Lightweight, non-weight-sharing student models for independent view representation.
Dual Cosine Embedding Loss for efficient knowledge transfer from a contrastive-trained teacher model.
Data Augmentation with Marginal Perturbations to improve pair-wise generalization.
Our approach achieves state-of-the-art performance with significant reductions in parameters (3×) and GFLOPs (13.5×) compared to existing models.

🚀 Features
Efficient Knowledge Transfer: A teacher-student framework eliminates the need for large batch sizes and hard-negative mining.
Dual Distillation: Uses cosine embedding loss to bring each student closer to the teacher's representations.
Lightweight Students: FasterViT-based architecture for resource-efficient training and inference.
Generalization Across Datasets: Achieves competitive results on CVUSA, CVACT, and VIGOR datasets.
🏆 Highlights
Parameters: Reduced 3× compared to the teacher model.
GFLOPs: Over 13.5× lower computational cost.
Performance: Competitive accuracy across datasets.
Scalability: Independent student training enables resource-efficient deployment.
📊 Results
CVUSA and CVACT Datasets
Method	R@1	R@5	R@10	R@1%
DSM	91.96%	97.50%	98.54%	99.67%
Sample4Geo	98.68%	99.68%	99.78%	99.87%
Distill4Geo	97.92%	99.58%	99.76%	99.86%
VIGOR Dataset (Cross-Split)
Method	R@1	R@5	R@10	Hit Rate
Sample4Geo	61.70%	83.50%	88.00%	69.87%
Distill4Geo	59.93%	83.09%	87.66%	68.14%
📂 Repository Structure
src/ - Core implementation of Distill4Geo, including teacher-student training modules.
datasets/ - Scripts for preparing and loading CVUSA, CVACT, and VIGOR datasets.
notebooks/ - Jupyter notebooks for result visualization and evaluation.
experiments/ - Configurations and logs for reproducibility.
📜 Citation
If you find our work useful, please cite:

bibtex
Copy code
@article{distill4geo2025,
  title={Distill4Geo: Streamlined Knowledge Transfer from Contrastive Weight-Sharing Teachers to Independent, Lightweight View Experts},
  author={Anonymous},
  journal={CVPR},
  year={2025}
}
💡 How to Contribute
We welcome contributions! Please refer to the CONTRIBUTING.md file for guidelines.

🔗 References
This repository builds upon prior works in cross-view geo-localization and knowledge distillation. See the paper for detailed references.

Feel free to star ⭐ this repository and follow for updates!
