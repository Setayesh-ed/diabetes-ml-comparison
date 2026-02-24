# Comparative Evaluation of Machine Learning Models on the Pima Indians Diabetes Dataset

This repository presents a Python-based pipeline for comparing multiple machine learning classifiers on the **Pima Indians Diabetes** dataset.  
The project follows a structured preprocessing and evaluation workflow inspired by academic research in medical data classification.

---

## Project Overview

The objective of this project is to predict the onset of diabetes based on diagnostic medical measurements.  
The implemented pipeline automates the following steps:

- **Data cleaning**: Handling missing or invalid values using median imputation.
- **Feature engineering**: Discretizing selected continuous variables into categorical bins.
- **Model comparison**: Evaluating multiple classifiers using stratified 10-fold cross-validation.
- **Model persistence**: Saving the best-performing model for future use.

---

## Models Implemented

The following machine learning models are implemented and evaluated:

- **Support Vector Machine (SVM)**  
- **Naive Bayes (Gaussian)**  
- **K-Nearest Neighbors (KNN)**  
- **Decision Tree** using entropy criterion (C4.5-like)  
- **Artificial Neural Network (MLP)**  

All models are trained and evaluated under identical preprocessing and validation conditions.

---

## Dataset

- **Name:** Pima Indians Diabetes Dataset  
- **Samples:** 768  
- **Features:** 8 numeric medical attributes  
- **Target variable:**  
  - `0` → Non-diabetic  
  - `1` → Diabetic  

Some medical attributes contain zero values that represent missing data; these values are handled during preprocessing.

---

## Preprocessing Pipeline

1. **Imputation**  
   Invalid zero values in selected medical features are replaced with the median of each feature.

2. **Discretization**  
   Continuous variables are converted into categorical bins:
   - Age → Young / Adult / Old  
   - Blood Pressure → Low / Normal / High  
   - BMI → Underweight / Normal / Overweight  
   - Glucose → Normal / High  

3. **Encoding**  
   Categorical features are encoded into numerical values to make them compatible with scikit-learn models.

4. **Validation Strategy**  
   Stratified 10-fold cross-validation is used to preserve class distribution across folds.

---

## Evaluation Metrics

Each model is evaluated using the following metrics:

- Accuracy  
- Precision  
- Recall  
- F1-score  

Reported results represent the mean performance across all cross-validation folds.

---

##  How to Run

### 1. Clone the repository

```bash
git clone https://github.com/Setayesh-ed/diabetes-ml-comparison.git
cd diabetes-ml-comparison

2. Install dependencies

pip install -r requirements.txt

3. Run the program

python compare_models_pima.py


---

 Project Structure

.
├── compare_models_pima.py   # Main implementation script
├── results_summary.csv      # Cross-validation results for all models (generated)
├── best_model.joblib        # Serialized best-performing model (generated)
├── README.md                # Project documentation
└── requirements.txt         # Project dependencies


---

 Results

After execution, a results_summary.csv file is generated containing evaluation metrics for all models.
The model with the highest accuracy is then trained on the full dataset and saved as best_model.joblib.


---

 Using the Saved Model

The trained model can be loaded and used for inference in another Python script:

import joblib

model = joblib.load("best_model.joblib")
# X_new_encoded must follow the same preprocessing and encoding steps
y_pred = model.predict(X_new_encoded)


---

 Notes

This project is intended for academic and educational purposes.

Results may vary depending on preprocessing choices and dataset characteristics.

The study highlights that model performance is highly dependent on feature representation and data distribution.
   
