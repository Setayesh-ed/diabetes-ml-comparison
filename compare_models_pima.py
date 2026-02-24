#!/usr/bin/env python3
"""
compare_models_pima.py

Compare several classifiers (SVM, Naive Bayes, KNN, Decision Tree (C4.5-like),
and MLP) on the Pima Indians Diabetes dataset. The script:
 - loads the dataset from a public URL,
 - imputes "0" values (common in Pima) with medians,
 - applies a discretization step inspired by the paper (Age/BP/BMI/Glucose),
 - encodes categorical features,
 - evaluates models with stratified 10-fold cross-validation,
 - saves results to CSV and dumps the best model with joblib.

Requirements:
  - python >= 3.8
  - pandas, numpy, scikit-learn, joblib
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

# -------------------------
# Configuration
# -------------------------
DATA_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
COLS = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'
]
# Columns in Pima where "0" indicates missing (commonly treated as missing)
MISSING_ZERO_COLS = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

RESULTS_CSV = Path("results_summary.csv")
BEST_MODEL_PKL = Path("best_model.joblib")
RANDOM_STATE = 42

# -------------------------
# Utility functions
# -------------------------
def load_and_preprocess(url: str = DATA_URL) -> pd.DataFrame:
    """Load Pima dataset from URL and impute zero-values for known columns."""
    df = pd.read_csv(url, names=COLS)
    # Replace 0 with NaN for selected columns, then impute median
    df[MISSING_ZERO_COLS] = df[MISSING_ZERO_COLS].replace(0, np.nan)
    imputer = SimpleImputer(strategy="median")
    df[MISSING_ZERO_COLS] = imputer.fit_transform(df[MISSING_ZERO_COLS])
    return df


def discretize_per_paper(df: pd.DataFrame) -> (pd.DataFrame, pd.Series):
    """
    Convert selected numeric columns to categorical bins following the approach
    in the paper (Age, BloodPressure, BMI, Glucose).
    Returns:
      X_cat: DataFrame of categorical features (as strings)
      y: target Series
    """
    df_copy = df.copy()

    df_copy['Age_Cat'] = pd.cut(df_copy['Age'], bins=[0, 25, 50, 150], labels=['Young', 'Adult', 'Old'])
    df_copy['BP_Cat'] = pd.cut(df_copy['BloodPressure'], bins=[0, 80, 120, 300], labels=['Low', 'Normal', 'High'])
    df_copy['BMI_Cat'] = pd.cut(df_copy['BMI'], bins=[0, 18.5, 25, 100], labels=['Underweight', 'Normal', 'Overweight'])
    df_copy['Glucose_Cat'] = pd.cut(df_copy['Glucose'], bins=[0, 140, 500], labels=['Normal', 'High'])

    selected = ['Age_Cat', 'BP_Cat', 'BMI_Cat', 'Glucose_Cat']
    X_cat = df_copy[selected].astype(str) # ensure dtype=object for encoder
    y = df_copy['Outcome'].astype(int)
    return X_cat, y


def encode_features(X_cat: pd.DataFrame) -> (pd.DataFrame, OrdinalEncoder):
    """
    Encode categorical columns into ordinal integers using OrdinalEncoder.
    Returns encoded DataFrame and the fitted encoder.
    """
    encoder = OrdinalEncoder()
    X_encoded = pd.DataFrame(encoder.fit_transform(X_cat), columns=X_cat.columns)
    return X_encoded, encoder


def build_models(random_state: int = RANDOM_STATE):
    """
    Build a dictionary of models to evaluate. Use pipelines for models that
    benefit from scaling.
    """
    models = {
        'SVM (RBF)': make_pipeline(StandardScaler(), SVC(kernel='rbf', C=1.0, probability=True, random_state=random_state)),
        'NaiveBayes (Gaussian)': GaussianNB(),
        'KNN (k=5)': make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
        # DecisionTree with entropy approximates C4.5 splitting criterion
        'DecisionTree (entropy)': DecisionTreeClassifier(criterion='entropy', random_state=random_state),
        'ANN (MLP)': make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000, random_state=random_state)
        )
    }
    return models


def evaluate_models(models: dict, X: pd.DataFrame, y: pd.Series, cv_splits: int = 10, random_state: int = RANDOM_STATE):
    """
    Evaluate all models with Stratified K-Fold cross-validation and return a DataFrame
    with mean and std of chosen metrics.
    """
    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    scoring = ['accuracy', 'precision', 'recall', 'f1']

    records = []
    for name, model in models.items():
        scores = cross_validate(model, X, y, cv=skf, scoring=scoring, n_jobs=-1, return_train_score=False)
        rec = {
            'model': name,
            'accuracy_mean': np.mean(scores['test_accuracy']),
            'accuracy_std': np.std(scores['test_accuracy']),
            'precision_mean': np.mean(scores['test_precision']),
            'precision_std': np.std(scores['test_precision']),
            'recall_mean': np.mean(scores['test_recall']),
            'recall_std': np.std(scores['test_recall']),
            'f1_mean': np.mean(scores['test_f1']),
            'f1_std': np.std(scores['test_f1']),
        }
        records.append(rec)
        # Print a compact summary to stdout
        print(f"{name:<25} | Acc: {rec['accuracy_mean']:.3f} ± {rec['accuracy_std']:.3f} | "
              f"Prec: {rec['precision_mean']:.3f} | Rec: {rec['recall_mean']:.3f} | F1: {rec['f1_mean']:.3f}")

    results_df = pd.DataFrame.from_records(records)
    results_df = results_df.sort_values(by='accuracy_mean', ascending=False).reset_index(drop=True)
    return results_df


def fit_and_save_best_model(models: dict, best_model_name: str, X: pd.DataFrame, y: pd.Series, out_path: Path = BEST_MODEL_PKL):
    """
    Fit the selected best model on the whole dataset and save to disk.
    Returns the fitted model.
    """
    best = models[best_model_name]
    best.fit(X, y)
    joblib.dump(best, out_path)
    return best


# -------------------------
# Main script
# -------------------------
def main():
    print("Loading and preprocessing dataset...")
    df = load_and_preprocess()

    print("Applying discretization (paper-like)...")
    X_cat, y = discretize_per_paper(df)

    print("Encoding categorical features...")
    X_encoded, encoder = encode_features(X_cat)

    print("Building model list...")
    models = build_models()

    print("\nEvaluating models with stratified 10-fold cross-validation (parallel)...\n")
    results_df = evaluate_models(models, X_encoded, y, cv_splits=10, random_state=RANDOM_STATE)

    print("\nSummary (sorted by accuracy):")
    print(results_df[['model', 'accuracy_mean', 'accuracy_std', 'precision_mean', 'recall_mean', 'f1_mean']].to_string(index=False))

    # Save results to CSV
    print(f"\nSaving results to {RESULTS_CSV}")
    results_df.to_csv(RESULTS_CSV, index=False)

    # Fit best model on whole data and save
    best_model_name = results_df.loc[0, 'model']
    print(f"\nFitting the best model ({best_model_name}) on the entire dataset and saving to {BEST_MODEL_PKL} ...")
    best_model = fit_and_save_best_model(models, best_model_name, X_encoded, y, out_path=BEST_MODEL_PKL)

    # Example: predict the first sample using the fitted best model
    sample = X_encoded.iloc[0:1]
    pred = best_model.predict(sample)
    print(f"\nSample prediction (first row) by best model '{best_model_name}': {int(pred[0])}")

    print("\nDone.")


if __name__ == "__main__":
    main()