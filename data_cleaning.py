import numpy as np
import pandas as pd

# Load DataSet

df=pd.read_csv('dataset/onlinefraud.csv')
# print(df.head())

print("Dataset Shape:", df.shape)
print("\nColumns:")
print(df.columns)


print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:", df.duplicated().sum())

df.drop(
    ["nameOrig", "nameDest", "isFlaggedFraud"],
    axis=1,
    inplace=True
)


print("\nColumns After Cleaning:")
print(df.columns)

df.to_csv("dataset/cleaned_onlinefraud.csv", index=False)

print("\nCleaned dataset saved successfully!")

clean_df = pd.read_csv("dataset/cleaned_onlinefraud.csv")

print(clean_df.head())

print(clean_df.columns)



"""
======================================
DATA CLEANING SUMMARY
======================================

Original Shape : (6362620, 11)

Missing Values : 0

Duplicate Rows : 0

Columns Removed :
✔ nameOrig
✔ nameDest
✔ isFlaggedFraud

Final Shape :
(6362620, 8)

Clean Dataset Saved :
dataset/cleaned_onlinefraud.csv

======================================
Data Cleaning Completed Successfully
======================================

"""