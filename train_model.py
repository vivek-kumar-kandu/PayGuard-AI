import pandas as pd
import numpy as np

df=pd.read_csv('dataset/cleaned_onlinefraud.csv')

# print(df.shape)
# print(df.head())

# Features Encoding  
# All columns are already numerical except the 'type' column,
# which is categorical. We use Label Encoding to convert it
# into numerical values.

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
df["type"] = encoder.fit_transform(df["type"])
# print(df.head())

'''
CASH_IN  0
CASH_OUT 1 
DEBIT    2
PAYMENT  3 
TRANSFER 4 

'''


#Seprating Features and Target Variables

X = df.drop("isFraud", axis=1)

y = df["isFraud"]

# Train-Test  Split  in 80% of 6,362,620 = 5,090,096 rows (approximately)  and 20% of 6,362,620 = 1,272,524 rows

# stratify=y ensures that the original class distribution
# (Legitimate vs Fraud) is preserved in both the training
# and testing datasets. This is important because the
# dataset is highly imbalanced.

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Handle Class Imbalance using SMOTE
from imblearn.over_sampling import SMOTE

# The dataset contains far fewer fraud transactions than
# legitimate transactions. SMOTE (Synthetic Minority
# Over-sampling Technique) generates synthetic fraud
# samples to balance the training dataset.
#
# NOTE:
# SMOTE is applied ONLY to the training data to prevent
# data leakage and ensure fair model evaluation.

smote = SMOTE(random_state=42)

X_train, y_train = smote.fit_resample(
    X_train,
    y_train
)

# Checking the class distribution after SMOTE
# print("\nClass Distribution After Applying SMOTE:")
# print(y_train.value_counts())

"""
Observation:

Before SMOTE:
Legitimate Transactions : 5,083,526
Fraud Transactions      : 6,570

Difference:
5,083,526 - 6,570 = 5,076,956

SMOTE generates 5,076,956 synthetic fraud samples
to balance the training dataset.

After SMOTE:
Legitimate : 5,083,526
Fraud      : 5,083,526

The training dataset is now balanced, enabling the
machine learning models to learn fraud patterns more
effectively.
"""


# print("\nTraining Data Shape :", X_train.shape)
# print("Testing Data Shape  :", X_test.shape)

print("\nData Preprocessing Completed Successfully!")

# Logistic Regression Model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score,precision_score,recall_score,f1_score,confusion_matrix,classification_report,)

# logistic_model = LogisticRegression(
#     random_state=42,
#     max_iter=1000
# )

# # # Train Model
# logistic_model.fit(X_train, y_train)

# # # Prediction
# y_pred = logistic_model.predict(X_test)

# Evaluation
# print("\n   Logistic Regression   ")
# print("\n=========================================")

# print("Accuracy :", accuracy_score(y_test, y_pred))
# print("Precision:", precision_score(y_test, y_pred))
# print("Recall   :", recall_score(y_test, y_pred))
# print("F1 Score :", f1_score(y_test, y_pred))

# print("\nConfusion Matrix")
# print(confusion_matrix(y_test, y_pred))

# print("\nClassification Report")
# print(classification_report(y_test, y_pred))

"""
Logistic Regression Results

Accuracy : 95.19%
Precision : 2.35%
Recall : 89.41%
F1 Score : 4.58%

Observation:
• Logistic Regression successfully detects most fraudulent transactions, achieving a high recall of 89.41%.
• However, the precision is very low because many legitimate transactions are incorrectly classified as fraud.
• This results in a high number of false positives.
• Logistic Regression provides a good baseline model but is not suitable as the final model for this project.
"""

# Decision Tree Classifier Model

from sklearn.tree import DecisionTreeClassifier

# Create Decision Tree Model
decision_tree = DecisionTreeClassifier(
    random_state=42
)

# Train Model
decision_tree.fit(X_train, y_train)

# Prediction
y_pred_dt = decision_tree.predict(X_test)


print("        Decision Tree Model")
print("========================================")

print("Accuracy :", accuracy_score(y_test, y_pred_dt))
print("Precision:", precision_score(y_test, y_pred_dt))
print("Recall   :", recall_score(y_test, y_pred_dt))
print("F1 Score :", f1_score(y_test, y_pred_dt))

print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred_dt))

print("\nClassification Report")
print(classification_report(y_test, y_pred_dt))

"""
Decision Tree Results

Accuracy : 99.94%
Precision : 70.84%
Recall : 96.10%
F1 Score : 81.56%

Observation:
• Decision Tree significantly outperformed Logistic Regression across all evaluation metrics.
• The model achieved excellent fraud detection performance with a Recall of 96.10%, successfully identifying most fraudulent transactions.
• Precision improved from 2.35% to 70.84%, greatly reducing false fraud alerts.
• The model achieved a balanced F1 Score of 81.56%, indicating strong overall classification performance.
• Decision Tree is a strong candidate for the final fraud detection system.
"""



# Random Forest Model
# ==========================================

# from sklearn.ensemble import RandomForestClassifier

# # Create Random Forest Model
# random_forest = RandomForestClassifier(
#     n_estimators=100,
#     random_state=42,
#     n_jobs=-1
# )

# # Train Model
# random_forest.fit(X_train, y_train)

# # Prediction
# y_pred_rf = random_forest.predict(X_test)

# print("        Random Forest Model")
# print("========================================")

# print("Accuracy :", accuracy_score(y_test, y_pred_rf))
# print("Precision:", precision_score(y_test, y_pred_rf))
# print("Recall   :", recall_score(y_test, y_pred_rf))
# print("F1 Score :", f1_score(y_test, y_pred_rf))

# print("\nConfusion Matrix")
# print(confusion_matrix(y_test, y_pred_rf))

# print("\nClassification Report")
# print(classification_report(y_test, y_pred_rf))

"""
Random Forest Results

Accuracy : 99.93%
Precision : 67.02%
Recall : 95.13%
F1 Score : 78.64%

Observation:
• Random Forest achieved excellent overall performance with an accuracy of 99.93%.
• The model detected most fraudulent transactions with a recall of 95.13%.
• Precision of 67.02% indicates that most predicted fraud transactions were correct.
• However, Decision Tree achieved slightly better Precision, Recall, and F1 Score.
• Therefore, Random Forest is not the best-performing model for this dataset.
"""


# XGBoost Model
# ==========================================

# ==========================================
# Create Training Subset for XGBoost
# ==========================================

# from sklearn.model_selection import train_test_split

# X_train_rf, _, y_train_rf, _ = train_test_split(
#     X_train,
#     y_train,
#     train_size=500000,
#     random_state=42,
#     stratify=y_train
# )

# print("\nXGBoost Training Shape:", X_train_rf.shape)
# from sklearn.metrics import (
#     accuracy_score,
#     precision_score,
#     recall_score,
#     f1_score,
#     confusion_matrix,
#     classification_report
# )

# # ==========================================
# # XGBoost Model
# # ==========================================

# from xgboost import XGBClassifier

# xgb_model = XGBClassifier(
#     n_estimators=100,
#     max_depth=6,
#     learning_rate=0.1,
#     random_state=42,
#     n_jobs=-1,
#     eval_metric="logloss",
#     tree_method="hist"
# )

# # Train Model
# xgb_model.fit(X_train_rf, y_train_rf)

# # Prediction
# y_pred_xgb = xgb_model.predict(X_test)

# print("\n========================================")
# print("          XGBoost Model")
# print("========================================")

# print("Accuracy :", accuracy_score(y_test, y_pred_xgb))
# print("Precision:", precision_score(y_test, y_pred_xgb))
# print("Recall   :", recall_score(y_test, y_pred_xgb))
# print("F1 Score :", f1_score(y_test, y_pred_xgb))

# print("\nConfusion Matrix")
# print(confusion_matrix(y_test, y_pred_xgb))

# print("\nClassification Report")
# print(classification_report(y_test, y_pred_xgb))

"""
XGBoost Results

Accuracy  : 99.38%
Precision : 17.12%
Recall    : 99.45%
F1 Score  : 29.22%

Observation:
• XGBoost achieved an excellent Recall of 99.45%, successfully detecting almost all fraudulent transactions.
• However, the Precision was only 17.12%, indicating that a large number of legitimate transactions were incorrectly classified as fraud.
• The low Precision resulted in a relatively low F1 Score of 29.22%.
• The model is highly sensitive to fraud detection but produces many false positive predictions.
• For this dataset, XGBoost did not outperform the Decision Tree or Random Forest models.

Conclusion:
• Although XGBoost achieved the highest Recall, its low Precision makes it less suitable for deployment in this project.
• Decision Tree remains the best-performing model because it provides the best balance between Precision, Recall, and F1 Score while maintaining excellent overall accuracy.
"""


#Final Model Comparison

"""
==========================================================
#Final Model Comparison
==========================================================

Model                   Accuracy    Precision    Recall     F1 Score
---------------------------------------------------------------------
Logistic Regression     95.19%       2.35%       89.41%      4.58%
Decision Tree           99.94%      70.84%       96.10%     81.56%
Random Forest           99.93%      67.02%       95.13%     78.64%
XGBoost                 99.38%      17.12%       99.45%     29.22%

Final Selected Model:
✔ Decision Tree Classifier

Reason:
• Highest F1 Score (81.56%)
• Highest Precision (70.84%)
• Excellent Recall (96.10%)
• Lowest False Positive Rate among all tested models.
• Best overall balance between fraud detection performance and reliability.
• Selected for deployment in the PayGuard AI web application.

==========================================================
Machine Learning Pipeline Completed Successfully
==========================================================
"""



# ==========================================
# Save Final Model and Label Encoder
# ==========================================

import joblib
import os

# Create model directory if it doesn't exist
os.makedirs("model", exist_ok=True)

# Save Decision Tree Model
joblib.dump(decision_tree, "model/fraud_detection_model.pkl")

# Save Label Encoder
joblib.dump(encoder, "model/label_encoder.pkl")

print("\n========================================")
print("Model Saved Successfully!")
print("========================================")
print("Decision Tree Model  : model/fraud_detection_model.pkl")
print("Label Encoder        : model/label_encoder.pkl")