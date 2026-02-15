\# Bank Marketing Classification - Streamlit App



\## 📌 Project Overview

This project demonstrates multiple machine learning classification models on the \*\*Bank Marketing Dataset\*\* from the UCI Machine Learning Repository.  

The task is to predict whether a client subscribes to a term deposit (`y` = yes/no) based on demographic and marketing campaign features.



\## 📝 Problem Statement

Financial institutions often run marketing campaigns to encourage customers to subscribe to term deposits.  

However, contacting every customer is costly and inefficient. The challenge is to build a classification model that can predict, based on customer and campaign attributes, whether a client will subscribe to a term deposit.  



By accurately identifying likely subscribers, banks can:

\- Improve campaign efficiency  

\- Reduce marketing costs  

\- Increase conversion rates  

\- Enhance customer targeting strategies  



This project evaluates multiple machine learning models to determine which performs best for this classification task.



\## 📊 Dataset Description

\- \*\*Source\*\*: \[UCI Machine Learning Repository – Bank Marketing Dataset](https://archive.ics.uci.edu/ml/datasets/bank+marketing)  

\- \*\*Instances\*\*: 45,211 (bank-additional-full.csv)  

\- \*\*Features\*\*: 16 input attributes (after one-hot encoding, feature count increases significantly)  

\- \*\*Target Variable\*\*: `y` (binary: `yes` = client subscribed, `no` = client did not subscribe)  



\### Key Features:

\- \*\*Demographic attributes\*\*: age, job, marital status, education  

\- \*\*Financial attributes\*\*: default, housing loan, personal loan  

\- \*\*Campaign-related attributes\*\*: contact type, month, day of week, duration, number of contacts performed, previous outcomes  

\- \*\*Social/economic context attributes\*\*: employment variation rate, consumer price index, consumer confidence index, euribor3m, number of employees

\### Comparison Table with the evaluation metrics:



| ML Model Name            | Accuracy | Precision | Recall | F1 Score |   MCC   |   AUC   |
|--------------------------|----------|-----------|--------|----------|---------|---------|
| Logistic Regression      | 0.9026   | 0.8885    | 0.9026 | 0.8896   | 0.4366  | 0.9025  |
| Decision Tree            | 0.9796   | 0.9799    | 0.9796 | 0.9798   | 0.9030  | 0.9572  |
| KNN                      | 0.9130   | 0.8616    | 0.9130 | 0.9037   | 0.5136  | 0.9361  |
| Naive Bayes              | 0.8534   | 0.8300    | 0.8534 | 0.8573   | 0.3309  | 0.7828  |
| Random Forest (Ensemble) | 0.9428   | 0.9428    | 0.9428 | 0.9360   | 0.6925  | 0.9676  |
| XGBoost (Ensemble)       | 0.9438   | 0.9408    | 0.9438 | 0.9410   | 0.7079  | 0.9722  |



\### Observation on model performance



| ML Model Name           | Observation about model performance                                                                 |
|-------------------------|-----------------------------------------------------------------------------------------------------|
| Logistic Regression     | Balanced performance with ~0.90 accuracy and AUC; moderate MCC (0.4366) shows limited handling of class imbalance despite solid recall. |
| Decision Tree           | Very high accuracy (~0.98) and strong MCC (0.9030); excellent precision/recall balance, though AUC slightly lower than ensembles. |
| KNN                     | Good accuracy (~0.91) and recall; MCC moderate (0.5136), indicating sensitivity to dataset imbalance and feature scaling. |
| Naive Bayes             | Lowest accuracy (~0.85) and weakest MCC (0.3309); struggles with complex feature interactions, though recall remains acceptable. |
| Random Forest (Ensemble)| Strong accuracy (~0.94) and high AUC (0.9676); MCC (0.6925) shows solid balance, but performance slightly below XGBoost. |
| XGBoost (Ensemble)      | Excellent accuracy (~0.94) with highest AUC (0.9722); MCC (0.7079) indicates robust balance between precision and recall, making it the most reliable overall. |



\### Dataset Notes:

\- Highly imbalanced (only ~11% positive class = subscribed).  

\- Requires preprocessing: one-hot encoding for categorical variables, scaling for numeric features.  

\- Useful for testing classification algorithms on imbalanced data.  



\## 📂 Project Structure

\- `app.py` → Streamlit app for interactive evaluation

\- `requirements.txt` → Dependencies for training and deployment

\- `README.md` → Documentation

\- `model/train\_models.py` → Training script for all models

\- `model/saved\_models/` → Pickled models and scaler





