# Heart Disease Predictor

A machine learning project that predicts the likelihood of heart disease in a patient based on clinical features. Built with Python, scikit-learn, and Gradio.

---

## Project Overview

Heart disease is one of the leading causes of death worldwide. Early detection plays a crucial role in improving patient outcomes. This project explores a clinical dataset of patient records, builds and evaluates several machine learning classifiers, and exposes the best-performing model through an interactive Gradio web interface.

The project covers the full ML workflow: exploratory data analysis (EDA), feature engineering, model training, evaluation, serialization, and deployment.

---

## Problem Statement

Given a set of clinical measurements for a patient — such as age, cholesterol level, resting blood pressure, and ECG results — predict whether the patient has heart disease (binary classification: 1 = Yes, 0 = No).

---

## Objectives

- Perform thorough exploratory data analysis on the heart disease dataset.
- Engineer features and preprocess data consistently for training and inference.
- Train and compare multiple ML classifiers (Random Forest, Logistic Regression, KNN).
- Serialize the best model and the fitted scaler for reproducible inference.
- Provide an interactive prediction interface via Gradio.

---

## Features

- **EDA**: Distribution plots, correlation heatmap, feature-vs-target boxplots.
- **Feature Engineering**: One-hot encoding of categorical features (`cp`, `thal`, `slope`), standard scaling of all features.
- **Model Comparison**: Random Forest, Logistic Regression, K-Nearest Neighbors.
- **Serialized Artifacts**: Pre-trained model (`heart_disease_model.pkl`) and scaler (`scaler.pkl`) included for immediate inference.
- **Interactive UI**: Gradio-powered web interface for real-time prediction, launched directly from the notebook.

---

## Technologies Used

| Category          | Library / Tool                  |
|-------------------|---------------------------------|
| Language          | Python 3.x                      |
| Data Manipulation | pandas, numpy                   |
| Visualization     | matplotlib, seaborn             |
| ML Framework      | scikit-learn                    |
| Model Persistence | joblib                          |
| UI / Demo         | Gradio                          |
| Notebook          | Jupyter                         |

---

## Project Structure

```
Heart-Disease-Predictor/
├── heart.csv                          # Dataset (1,025 patient records)
├── heart_disease_model.pkl            # Trained Random Forest model
├── scaler.pkl                         # Fitted StandardScaler
├── Untitled.ipynb                     # Original notebook (root)
├── requirements.txt                   # Python dependencies
├── README.md                          # Project documentation
├── .gitignore                         # Git ignore rules
└── notebooks/
    └── heart_disease_predictor.ipynb  # Annotated notebook (EDA → model → UI)
```

---

## Dataset Description

| Property        | Detail                            |
|-----------------|-----------------------------------|
| Source          | UCI Heart Disease Dataset (Cleveland subset) |
| Rows            | 1,025 patient records             |
| Columns         | 14 (13 features + 1 target)       |
| Missing values  | None                              |
| Target          | `target` — 1 (disease), 0 (no disease) |
| Class balance   | 526 positive / 499 negative       |

### Feature Descriptions

| Feature    | Description                                                          |
|------------|----------------------------------------------------------------------|
| `age`      | Age of the patient in years                                          |
| `sex`      | Sex (1 = male, 0 = female)                                          |
| `cp`       | Chest pain type (0–3)                                                |
| `trestbps` | Resting blood pressure (mm Hg)                                       |
| `chol`     | Serum cholesterol (mg/dl)                                            |
| `fbs`      | Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)               |
| `restecg`  | Resting ECG results (0–2)                                            |
| `thalach`  | Maximum heart rate achieved                                          |
| `exang`    | Exercise-induced angina (1 = yes, 0 = no)                           |
| `oldpeak`  | ST depression induced by exercise relative to rest                   |
| `slope`    | Slope of the peak exercise ST segment (0–2)                          |
| `ca`       | Number of major vessels colored by fluoroscopy (0–3)                 |
| `thal`     | Thalassemia type (0–3)                                               |
| `target`   | Heart disease present (1 = yes, 0 = no) — **prediction target**     |

---

## Machine Learning Workflow

```
heart.csv
    │
    ▼
Exploratory Data Analysis
    │  - Shape, dtypes, missing value check
    │  - Target class distribution
    │  - Correlation heatmap
    │  - Feature distribution histograms
    │  - Boxplots per feature vs. target
    │
    ▼
Feature Engineering
    │  - One-hot encode: cp, thal, slope (drop_first=True)
    │  - Result: 18 input features
    │
    ▼
Preprocessing
    │  - StandardScaler fitted on training data
    │
    ▼
Train / Test Split (80% / 20%, random_state=42)
    │
    ▼
Model Training & Evaluation
    │  - Random Forest Classifier  ← selected model
    │  - Logistic Regression
    │  - K-Nearest Neighbors
    │
    ▼
Serialization
    │  - heart_disease_model.pkl (Random Forest)
    │  - scaler.pkl (StandardScaler)
    │
    ▼
Gradio UI (interactive prediction interface)
```

---

## Model Training & Evaluation Results

| Model                  | Test Accuracy |
|------------------------|---------------|
| Random Forest          | **1.00**      |
| K-Nearest Neighbors    | 0.84          |
| Logistic Regression    | 0.82          |

> **Note:** The Random Forest model achieved 1.00 accuracy on the test set. This is a result observed during training on this particular dataset and split. Real-world performance may vary. Always validate on independent, unseen data before deployment.

The selected model (Random Forest) was serialized with `joblib` and saved as `heart_disease_model.pkl`.

---

## Installation

### Prerequisites

- Python 3.8 or higher
- pip

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/GaureshShinde25/Heart-Disease-Predictor.git
cd Heart-Disease-Predictor

# 2. (Optional) Create and activate a virtual environment
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## How to Run

### Option 1 — Run the Notebook (Recommended)

Open the notebook and run all cells to reproduce the full EDA, training, and UI:

```bash
jupyter notebook notebooks/heart_disease_predictor.ipynb
```

The last cell launches a Gradio interface locally at `http://127.0.0.1:7860`.

### Option 2 — Use the Pre-trained Model Directly

```python
import numpy as np
import joblib

model = joblib.load('heart_disease_model.pkl')
scaler = joblib.load('scaler.pkl')

# Example: one patient record
# Features: age, sex, trestbps, chol, fbs, restecg, thalach, exang, oldpeak,
#           ca, cp_1, cp_2, cp_3, thal_1, thal_2, thal_3, slope_1, slope_2
sample = np.array([[52, 1, 125, 212, 0, 1, 168, 0, 1.0, 2, 0, 0, 0, 0, 0, 1, 0, 1]])
sample_scaled = scaler.transform(sample)
prediction = model.predict(sample_scaled)[0]

print("Heart Disease Detected!" if prediction == 1 else "No Heart Disease Detected")
```

---

## Example Prediction Workflow

1. Open the Gradio UI (launched from the notebook's final cell).
2. Enter patient data:
   - Numeric fields: `Age`, `Resting Blood Pressure`, `Cholesterol`, `Maximum Heart Rate`, `Oldpeak`
   - Binary radio fields: `Sex`, `Fasting Blood Sugar`, `Exercise-induced Angina`
   - Multi-class radio: `Resting ECG`, `Number of Major Vessels`
   - Checkbox fields for one-hot encoded features: `Chest Pain Type` (cp_1/2/3), `Thal` (1/2/3), `Slope` (1/2)
3. Click **Submit** to receive a prediction: `"Heart Disease Detected!"` or `"No Heart Disease Detected"`.

---

## Future Improvements

- **Better feature encoding UI**: Replace raw one-hot checkboxes in Gradio with user-friendly dropdowns for `cp`, `thal`, and `slope`.
- **Cross-validation**: Add k-fold CV for more robust model evaluation.
- **Hyperparameter tuning**: Use `GridSearchCV` or `RandomizedSearchCV` to optimize the Random Forest.
- **SHAP explanations**: Add feature importance / explainability output to the UI.
- **Deployment**: Package as a standalone web app (FastAPI + Docker) for cloud deployment.
- **Additional models**: Evaluate XGBoost, LightGBM, or ensemble stacking.

---

## Medical Disclaimer

> **This project is intended for educational and research purposes only.**
>
> The predictions generated by this model are **not a substitute for professional medical advice, diagnosis, or treatment**. The model has been trained on a limited dataset and has not been clinically validated. Do not use the output of this tool to make real healthcare decisions.
>
> If you have concerns about your heart health, please consult a qualified healthcare professional.

---

## License

This project is open source and available under the [MIT License](LICENSE).
