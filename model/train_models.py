# model/train_models.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import pickle
import os

# Load Bank Marketing dataset (UCI ML Repository)
df = pd.read_csv("bank-full.csv", sep=";")

# Target column
y = df["y"].map({"yes": 1, "no": 0})  # binary classification
X = df.drop("y", axis=1)

# One-hot encode categorical features
X = pd.get_dummies(X, drop_first=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save scaler
os.makedirs("model/saved_models", exist_ok=True)
with open("model/saved_models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Define models
models = {
    "Logistic_Regression.pkl": LogisticRegression(max_iter=1000),
    "Decision_Tree.pkl": DecisionTreeClassifier(),
    "KNN.pkl": KNeighborsClassifier(),
    "Naive_Bayes.pkl": GaussianNB(),
    "Random_Forest.pkl": RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42),
    "XGBoost.pkl": xgb.XGBClassifier(eval_metric="logloss")
}

# Train and save models
for filename, model in models.items():
    model.fit(X_train, y_train)
    with open(f"model/saved_models/{filename}", "wb") as f:
        pickle.dump(model, f)


print("Training complete. Models and scaler saved in model/saved_models/")
