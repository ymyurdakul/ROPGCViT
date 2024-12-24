# ROPGCViT
![image](https://github.com/user-attachments/assets/a70c57f3-8d27-47cd-815b-e90f5b70f9d1)
# ROPGCViT: An Advanced Global Context Vision Transformer for High-Precision Retinopathy of Prematurity Diagnosis

This repository contains the implementation of ROPGCViT, a novel Vision Transformer architecture designed for the diagnosis of Retinopathy of Prematurity (ROP). ROPGCViT enhances the GCViT architecture with Squeeze-and-Excitation (SE) blocks and Residual Multi-Layer Perceptrons (RMLPs) to achieve superior accuracy and robustness.

## Key Features

- **SE Block Integration:** Enhances the stem block to recalibrate channel-based feature representations.
- **Residual MLP Layers:** Introduces skip connections to improve learning dynamics and prevent gradient vanishing.
- **High Classification Performance:** Achieves state-of-the-art results for ROP diagnosis, with 94.69% accuracy and 94.84% precision.

## Dataset

The ROP dataset includes 1099 fundus images classified into five categories: Normal, Stage 1, Stage 2, Stage 3, and Laser scars. The dataset was collected over 19 years and can be accessed [here](https://figshare.com/articles/figure/_b_A_Fundus_Image_Dataset_for_Intelligent_b_b_Retinopathy_of_Prematurity_b_b_System_b_/25514449).

| Category      | Training | Validation | Testing | Total |
|---------------|----------|------------|---------|-------|
| Normal        | 188      | 24         | 24      | 236   |
| Stage 1       | 75       | 9          | 10      | 94    |
| Stage 2       | 132      | 16         | 17      | 165   |
| Stage 3       | 208      | 26         | 27      | 261   |
| Laser Scars   | 274      | 34         | 35      | 343   |
| **Total**     | 877      | 109        | 113     | 1099  |

## Results

The ROPGCViT model demonstrated significant improvements over 50 tested deep learning models and existing literature:

| Model        | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | Kappa (%) |
|--------------|--------------|----------------|-------------|--------------|-----------|
| GCViT        | 92.03        | 93.39         | 92.03      | 92.32       | 89.72     |
| GCViT + SE   | 93.80        | 94.15         | 93.80      | 93.67       | 91.96     |
| **ROPGCViT** | **94.69**    | **94.84**     | **94.69**  | **94.60**   | **93.10** |

## Dependencies

- Python 3.11.x
- TensorFlow 2.14.0
- Keras 2.11.4
- CUDA 12.7

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/ROPGCViT.git
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Authors
- **Şakir Taşdemir**  
  Selçuk University, Computer Engineering Department  
  [ORCID](https://orcid.org/0000-0002-2433-246X) | Email: stasdemir@selcuk.edu.tr
  
  - **Kübra Uyar**  
  Alanya Alaaddin Keykubat University, Computer Engineering Department  
  [ORCID](https://orcid.org/0000-0001-5345-3319) | Email: kubra.uyar@alanya.edu.tr

- **Mustafa Yurdakul**  
  Kırıkkale University, Computer Engineering Department  
  [ORCID](https://orcid.org/0000-0003-0562-4931) | Email: mustafayurdakul@kku.edu.tr  

## Contact

For questions or feedback, please contact:  
**Mustafa Yurdakul**: mustafayurdakul@kku.edu.tr  



