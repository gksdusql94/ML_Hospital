# Project Name: Diabetes Readmission Prediction

## Overview
This project aims to predict whether a patient will be readmitted to the hospital based on various clinical, demographic, and text-based data. The dataset consists of 8,000 patient records, and the goal is to build a robust machine learning model that can accurately predict the likelihood of hospital readmission.

## Table of Contents
1. [Data Preprocessing](#data-preprocessing)
2. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
3. [Model Building](#model-building)
4. [Text Data Handling with TF-IDF](#text-data-handling-with-tf-idf)
5. [Model Stacking and Final Prediction](#model-stacking-and-final-prediction)
6. [Results and ROC Curve Visualization](#results-and-roc-curve-visualization)
7. [Correlation Analysis with Heatmap](#correlation-analysis-with-heatmap)
8. [Conclusion](#conclusion)
9. [Future Work](#future-work)

---

## 1. Data Preprocessing
  Before building the model, we performed the following preprocessing steps:
  - Columns with excessive missing values (e.g., `max_glu_serum`) were removed.
  - Categorical columns were encoded using Label Encoding and One-Hot Encoding.
  - Numerical columns were standardized using `StandardScaler` to normalize the data.
  - Binary columns like `change` and `diabetesMed` were mapped to `0` (No) and `1` (Yes).
  By handling missing data and encoding categorical variables, the dataset was prepared for machine learning model training.

      ```python
      import pandas as pd
      import numpy as np
      from sklearn.preprocessing import LabelEncoder, StandardScaler
      
      df.replace("?", np.nan, inplace=True)
      
      df['discharge_disposition_id'].fillna('Unknown', inplace=True)
      df['admission_source_id'].fillna('Not Mapped', inplace=True)
      df['medical_specialty'].fillna('Unknown', inplace=True)
      df['payer_code'].fillna('Unknown', inplace=True)
      
      df.drop(columns=['max_glu_serum'], inplace=True)
      
      le = LabelEncoder()
      df['discharge_disposition_id'] = le.fit_transform(df['discharge_disposition_id'])
      df['admission_source_id'] = le.fit_transform(df['admission_source_id'])
      df['payer_code'] = le.fit_transform(df['payer_code'])
      df['medical_specialty'] = le.fit_transform(df['medical_specialty'])
      
      scaler = StandardScaler()
      df_numeric = df.select_dtypes(include=np.number)
      df_numeric_scaled = scaler.fit_transform(df_numeric)

## 2. Exploratory Data Analysis (EDA)
  We conducted a detailed analysis of the dataset to understand the distribution of the target variable and to identify correlations between numerical features. Visualizations like bar plots and heatmaps helped in this process.
      
      ```python
        import matplotlib.pyplot as plt
        import seaborn as sns
        plt.figure(figsize=(6, 4)) # Visualize the target variable distribution
        sns.countplot(x='readmitted', data=df)
        plt.title('Distribution of Readmitted Patients')
        plt.xlabel('Readmitted')
        plt.ylabel('Count')
        plt.show()


