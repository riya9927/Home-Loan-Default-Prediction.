# Home-Loan-Default-Prediction.(Deep Learning Project)



This project aims to predict whether a loan applicant will default on their loan using historical loan data. A deep learning model is built with Keras and TensorFlow to handle this binary classification problem, which is also highly imbalanced.

---

## 📌 Problem Statement

For a safe and secure lending experience, it’s important to analyze past data. The goal of this project is to develop a prediction model that identifies potential defaulters based on their application data.

---

## 🎯 Objective

Build a deep learning model that:
- Predicts whether or not a loan applicant will default
- Handles imbalanced dataset using SMOTE
- Evaluates performance using **AUC-ROC** and **Sensitivity**

---

## 📁 Dataset

- `loan_data.csv`: Main dataset containing applicant data and loan status (`TARGET`)
   Data Set: https://drive.google.com/file/d/1FpNO9wykWbVd2mDdp1Bpzu9VeXBlK8vE/view?usp=sharing
- `Data_Dictionary.csv`: Description of the dataset features

---

## 🧪 Tech Stack

- Python
- TensorFlow & Keras
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- imbalanced-learn (SMOTE)

---

## ⚙️ Steps Performed

1. **Load and Explore Dataset**
2. **Handle Missing Values**
3. **EDA and Target Distribution Analysis**
4. **Balance Dataset using SMOTE**
5. **Label Encoding and Feature Scaling**
6. **Build Deep Learning Model**
7. **Use Callbacks (EarlyStopping, ReduceLROnPlateau)**
8. **Evaluate using AUC-ROC and Sensitivity**
9. **Visualize ROC Curve**
    
---

## 🧠 Model Architecture

- Dense(128) + BatchNormalization + Dropout(0.4)
- Dense(64) + BatchNormalization + Dropout(0.3)
- Dense(32) + Dropout(0.3)
- Output: Dense(1, activation='sigmoid')

---

## 📊 Model Performance

| Metric        | Score   |
|---------------|---------|
| Sensitivity      | ~**0.8724** High (based on confusion matrix)|
| AUC-ROC       | **0.9217** |

---

## 📈 Visuals

- Distribution plots of `TARGET`
- Balanced vs. Imbalanced comparison
- ROC Curve

---

## 🚀 Future Improvements

- Hyperparameter tuning with GridSearchCV
- Ensemble models with XGBoost or LightGBM
- Model deployment with Streamlit/Flask
- Feature selection and dimensionality reduction

