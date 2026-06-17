# Applied-Machine-Learning
A comprehensive Machine Learning pipeline for Credit Risk Prediction using Python. The project includes EDA, feature engineering, SMOTE oversampling, hyperparameter tuning, and comparison of Logistic Regression, SVM, Random Forest, LightGBM, and XGBoost models for binary credit risk classification.

# Credit Risk Prediction using Machine Learning

This project was developed as part of the Applied Machine Learning course in the Master of Data Science and Business Analytics program.

The objective of this project is to predict whether a borrower is a low-risk or high-risk applicant using a structured credit risk dataset. A complete machine learning pipeline was designed, including:

- Exploratory Data Analysis (EDA)
- Data Cleaning and Missing Value Imputation
- Feature Engineering
- One-Hot Encoding
- Power Transformation and Robust Scaling
- SMOTE for Class Imbalance Handling
- Hyperparameter Tuning with Cross-Validation
- Model Evaluation using Accuracy, Precision, Recall, F1-Score, and Confusion Matrix

## Models Implemented
- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest
- LightGBM
- XGBoost

## Key Results
- LightGBM achieved the highest accuracy: **93.81%**
- XGBoost achieved **93.32%** accuracy with strong recall for identifying risky borrowers.
- Random Forest achieved **93.16%** accuracy with excellent overall performance.
- Ensemble and boosting models significantly outperformed linear models for this credit risk classification task.

## Tools & Libraries
- Python
- Pandas & NumPy
- Scikit-learn
- Imbalanced-learn (SMOTE)
- LightGBM
- XGBoost
- Matplotlib & Seaborn
- Google Colab

This project demonstrates an end-to-end machine learning workflow for financial risk prediction and highlights the effectiveness of ensemble learning methods for structured tabular data.
