# 🚑 SepsisSense: Early ICU Sepsis Prediction using Temporal Machine Learning

> An end-to-end Healthcare AI system designed to predict early sepsis risk using real-world ICU patient time-series data and clinically informed machine learning pipelines.

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge\&logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-ML_Model-orange?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed_App-red?style=for-the-badge\&logo=streamlit)
![Healthcare AI](https://img.shields.io/badge/Domain-Healthcare_AI-green?style=for-the-badge)
![Status](https://img.shields.io/badge/Project-Production_Ready-success?style=for-the-badge)

---

# 📌 Problem Statement

Sepsis is one of the leading causes of mortality in Intensive Care Units (ICUs). Delayed detection can rapidly result in:

* Organ failure
* Septic shock
* Multi-organ dysfunction
* Increased ICU mortality

Traditional diagnosis often occurs after significant physiological deterioration. This project focuses on leveraging Machine Learning and temporal clinical data to assist in identifying early warning signs of sepsis before critical deterioration occurs.

---
#🔗WEBSITE
https://sepsissense.streamlit.app/
Please,do checkout!

---

# 🧠 Project Overview

SepsisSense was developed using a large-scale real-world ICU dataset containing over **1.5 million patient records** with temporal physiological measurements such as:

* Heart Rate (HR)
* Blood Pressure (MAP/SBP/DBP)
* Respiratory Rate
* Oxygen Saturation
* Laboratory Biomarkers
* ICU Progression Data

The system performs:

* Leakage-safe patient-wise preprocessing
* Temporal feature engineering
* Missing value handling for sparse ICU data
* Imbalance-aware classification using XGBoost
* Real-time risk prediction through Streamlit deployment

---

# ⚙️ Key Features

## ✅ Temporal Healthcare Preprocessing

* Patient-wise interpolation
* Forward filling of clinical measurements
* Leakage-safe GroupShuffleSplit validation
* Sparse feature handling for ICU lab tests

## ✅ Clinical Feature Engineering

Engineered medically meaningful features such as:

* Shock Index
* Pulse Pressure
* Rolling Mean Statistics
* Physiological Trend Differencing
* HR × Temperature Interaction

## ✅ Imbalanced Learning Strategy

* Handled severe class imbalance using:

  * `scale_pos_weight`
  * Recall-oriented evaluation
  * Threshold optimization

## ✅ Real-Time Prediction Ready

* Streamlit-powered interactive dashboard
* Live sepsis risk probability prediction
* Clinical decision support style output

---

# 📊 Model Performance

| Metric           | Score                        |
| ---------------- | ---------------------------- |
| ROC-AUC          | **0.825**                    |
| Sepsis Recall    | **68%**                      |
| Optimized Recall | **90%**                      |
| Dataset Size     | **~1.5 Million ICU Records** |

### 📌 Clinical Interpretation

The model prioritizes **high recall** to reduce the probability of missing septic patients, which is critical in real-world healthcare scenarios where delayed intervention can be life-threatening.

---

# 🧪 Tech Stack

```bash
Python
Pandas
NumPy
Scikit-learn
XGBoost
Matplotlib
Streamlit
```

---

# 📁 Project Structure

```bash
SepsisSense/
│
├── app.py
├── preprocessing.py
├── model_1.py
├── EDA_analysis.ipynb
├── requirements.txt
├── README.md
├── LICENSE
└── images/
```

---

# 📷 Project Visualizations

## 🖥 Streamlit Dashboard


## 📈 ROC Curve

(Add ROC curve image here)

## 📊 Feature Importance

(Add feature importance graph here)

## 🧾 Confusion Matrix

(Add confusion matrix image here)

---

# 🚀 Installation

```bash
git clone https://github.com/ManvithMogaveera/Sepsis-Early-Detection-using-Machine-Learning-on-Real-ICU-Patient-Data.git

cd SepsisSense

pip install -r requirements.txt
```

---

# ▶️ Run the Application

```bash
streamlit run app.py
```

---

# 🔬 Dataset

Dataset used:
**PhysioNet / Computing in Cardiology Sepsis Challenge Dataset**

Due to dataset size and licensing constraints, the original dataset is not uploaded to this repository.

---

# 🌍 Real-World Impact

This project demonstrates how Machine Learning can assist healthcare professionals in:

* Early critical care risk detection
* Continuous ICU patient monitoring
* Clinical decision support systems
* Reducing delayed sepsis diagnosis

The pipeline was intentionally designed with healthcare-safe preprocessing and leakage prevention techniques to better simulate real-world ICU deployment scenarios.

---

# 📌 Future Improvements

* SHAP Explainability Integration
* Real-Time ICU Monitoring Support
* Transformer/LSTM Temporal Models
* Cloud Deployment
* Live Clinical Alert System

---

# 👨‍💻 Author

### Manvith Mogaveera

AI & Machine Learning Enthusiast focused on:

* Healthcare AI
* Clinical ML Systems
* Real-World Intelligent Applications
* Temporal Machine Learning

---

⭐ If you found this project interesting, consider starring the repository.
