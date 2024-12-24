# 🌟 ROPGCViT: Advanced Vision Transformer for Retinopathy Diagnosis

![image]\(https\://github.com/user-attachments/assets/4cd20610-9535-4856-b774-64c8b7c43211)

Welcome to the **ROPGCViT** repository{: .img-circle .img-small} with unparalleled precision. By leveraging **Squeeze-and-Excitation (SE) blocks** and **Residual Multi-Layer Perceptrons (RMLPs)**, ROPGCViT sets new standards in medical imaging.

---

## ✨ Highlights

### 🚀 Key Features

- 🧠 **SE Block Integration:** Enhances feature representation by adaptively recalibrating channels.
- 🔗 **Residual MLP Layers:** Boosts learning dynamics and prevents gradient vanishing with skip connections.
- 📊 **Record-Breaking Performance:** Achieves **94.69% accuracy** and **94.84% precision**, outperforming existing benchmarks.

### 🎯 Why ROPGCViT?

- Seamlessly combines **local** and **global context awareness**.
- Designed to address complex **multi-stage ROP classification**.
- Efficient and robust, tailored for real-world medical datasets.

---

## 📂 Dataset Overview

The **ROP dataset** includes 1,099 fundus images classified into five distinct categories:

- **Normal**
- **Stage 1**
- **Stage 2**
- **Stage 3**
- **Laser scars**

Collected over 19 years, this dataset serves as a foundation for advancing ROP diagnosis. Access it [here](https://figshare.com/articles/figure/_b_A_Fundus_Image_Dataset_for_Intelligent_b_b_Retinopathy_of_Prematurity_b_b_System_b_/25514449).

| **Category** | **Training** | **Validation** | **Testing** | **Total** |
| ------------ | ------------ | -------------- | ----------- | --------- |
| Normal       | 188          | 24             | 24          | 236       |
| Stage 1      | 75           | 9              | 10          | 94        |
| Stage 2      | 132          | 16             | 17          | 165       |
| Stage 3      | 208          | 26             | 27          | 261       |
| Laser Scars  | 274          | 34             | 35          | 343       |
| **Total**    | **877**      | **109**        | **113**     | **1099**  |

---

## 📈 Results & Performance

ROPGCViT surpasses 50 tested deep learning models and the state-of-the-art literature:

| **Model**    | **Accuracy (%)** | **Precision (%)** | **Recall (%)** | **F1-Score (%)** | **Kappa (%)** |
| ------------ | ---------------- | ----------------- | -------------- | ---------------- | ------------- |
| GCViT        | 92.03            | 93.39             | 92.03          | 92.32            | 89.72         |
| GCViT + SE   | 93.80            | 94.15             | 93.80          | 93.67            | 91.96         |
| **ROPGCViT** | **94.69**        | **94.84**         | **94.69**      | **94.60**        | **93.10**     |



## 🛠️ Dependencies

Ensure the following are installed:

- **Python**: 3.11.x
- **TensorFlow**: 2.14.0
- **Keras**: 2.11.4
- **CUDA**: 12.7

---

## 💻 Quick Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/ROPGCViT.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## 👩‍🔬 Meet the Team

### 🧑‍🏫 Prof. Dr. Şakir Taşdemir



- **Affiliation**: [Selçuk University, Computer Engineering Department](https://www.selcuk.edu.tr/)
- **Expertise**: Deep learning, Vision Transformers, Medical Imaging
- 🌍 **Location**: [Konya, Turkey](https://goo.gl/maps/5fUZKKovDfAQzdPXA)
- [ORCID](https://orcid.org/0000-0002-2433-246X)
- 📧 **Email**: [stasdemir@selcuk.edu.tr](mailto\:stasdemir@selcuk.edu.tr)

### 👩‍🏫 Assist. Prof. Dr. Kübra Uyar



- **Affiliation**: [Alanya Alaaddin Keykubat University, Computer Engineering Department](https://www.alanya.edu.tr/)
- **Expertise**: Machine Learning, Computer Vision, Hybrid Models
- 🌍 **Location**: [Antalya, Turkey](https://goo.gl/maps/KdRG6E8FBCyE6EGT8)
- [ORCID](https://orcid.org/0000-0001-5345-3319)
- 📧 **Email**: [kubra.uyar@alanya.edu.tr](mailto\:kubra.uyar@alanya.edu.tr)

### 🧑‍🎓 Mustafa Yurdakul (PhD Candidate)



- **Affiliation**: [Kırıkkale University, Computer Engineering Department](https://www.kku.edu.tr/)
- **Expertise**: Vision Transformers, Attention Mechanisms, AI in Healthcare
- 🌍 **Location**: [Kırıkkale, Turkey](https://goo.gl/maps/BsMkq2RtFbDyzXyZ7)
- 📧 **Email**: [mustafayurdakul@kku.edu.tr](mailto\:mustafayurdakul@kku.edu.tr)

---

## 📬 Contact Us

For inquiries, feedback, or collaboration, reach out to:
📧 **Mustafa Yurdakul**: [mustafayurdakul@kku.edu.tr](mailto\:mustafayurdakul@kku.edu.tr)

---

## 📜 License

This project is licensed under the **MIT License**. See the LICENSE file for details.

---

## 🌐 Join the Community

Stay updated with our latest research, tools, and insights in deep learning for medical applications. Follow us and contribute to the future of AI in healthcare!

