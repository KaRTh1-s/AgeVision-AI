# 🔮 AgeVision AI — High-Precision Facial Age & Gender Estimation

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0%2Bcu124-EE4C2C.svg)](https://pytorch.org/)
[![Backbone](https://img.shields.io/badge/Backbone-ConvNeXt--Small-green.svg)](https://github.com/facebookresearch/ConvNeXt)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end deep learning system for high-precision human age and gender estimation from portrait photos using **Deep Label Distribution Learning (DLDL)** and modern **ConvNeXt** vision backbones.

---

## 🌟 Key Features

* **Continuous Age Distribution (DLDL):** Represents human age as a 101-class Gaussian probability distribution ($\sigma = 2.0$) optimized via KL-Divergence and Mean-Variance regularized loss.
* **Modern Vision Backbone:** Powered by `convnext_small.fb_in22k` pre-trained on ImageNet-22K (~14M images).
* **Pre-Inference Quality Gate (`validator.py`):**
  * **Face Verification:** MTCNN deep detector rejects non-faces, animals, and cartoons ($p < 0.90$).
  * **Sharpness Gate:** Laplacian variance filter rejects motion blur ($\sigma^2 < 100$).
  * **Pose Profile Check:** Facial landmark geometry check rejects extreme side-profile angles.
  * **Resolution Check:** Rejects low-res crops smaller than $80 \times 80\text{ px}$.
* **Multi-Attribute Inference:** Simultaneous Age, Age Group, Confidence Interval, and Gender prediction via parallel Hugging Face Vision Transformer.
* **Test-Time Augmentation (TTA):** Blends original and flipped probability distributions for enhanced robustness.
* **Production Web UI:** Glassmorphic dark-mode Flask web dashboard with live animated indicators.

---

## 📊 Benchmark Results

| Model Architecture | Loss Strategy | Validation MAE | Validation RMSE |
|---|---|---|---|
| Baseline MobileNetV2 | Plain MSE Regression | 13.52 yrs | 17.84 yrs |
| EfficientNet-B4 | Gaussian DLDL | 6.53 yrs | 8.81 yrs |
| **ConvNeXt-Small (V3)** | **Mean-Variance DLDL + TTA** | **5.59 yrs** | **7.91 yrs** |

*Trained on 375,775 verified face images (309,462 Train / 66,313 Val).*

---

## 📁 Repository Structure

```
├── app.py                     # Flask web application & REST API
├── inference.py               # Inference engine with TTA, uncertainty & gender branch
├── validator.py               # Pre-inference safety & quality gate (MTCNN + Blur)
├── training/
│   └── train_age_model.py     # Complete PyTorch training pipeline with AMP & Cosine LR
├── models/
│   └── best_age_model.pt      # Trained model weights (Download from GitHub Releases)
├── results/
│   └── final_metrics.json     # Quantitative evaluation metrics
├── requirements_training.txt  # Python package dependencies
└── Project_Documentation_Report.md  # Comprehensive technical documentation & interview guide
```

---

## 🚀 Quick Start Guide

### 1. Clone the Repository
```bash
git clone https://github.com/KaRTh1-s/AgeVision-AI.git
cd AgeVision-AI
```

### 2. Set Up Virtual Environment & Dependencies
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

pip install -r requirements_training.txt
pip install timm facenet-pytorch transformers
```

### 3. Download Trained Model Weights
Download `best_age_model.pt` from the [GitHub Releases](https://github.com/KaRTh1-s/AgeVision-AI/releases) section and place it inside the `models/` directory:
```
models/best_age_model.pt
```

### 4. Run the Web Application
```bash
python app.py
```
Open your browser at **`http://localhost:8080`**.

---

## 🧠 Loss Function Formulation

$$\mathcal{L}_{total} = \mathcal{L}_{KL}(p \parallel \hat{p}) + \lambda_1 \mathcal{L}_{L1}(E[\hat{p}], y) + \lambda_2 \text{Var}(\hat{p})$$

* **KL Divergence:** Shapes the continuous Gaussian probability curve over 101 age bins.
* **L1 Expectation:** Directly penalizes distance between the expected prediction $\hat{y} = \sum i \cdot \hat{p}_i$ and true age $y$.
* **Variance Regularization:** Forces the probability distribution to form a sharp, confident peak.

---

## 📄 License
This project is licensed under the MIT License.
