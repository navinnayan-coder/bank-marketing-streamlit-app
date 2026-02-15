import streamlit as st_lit
import pandas as pd
import pickle
import os
import requests
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)
import seaborn as sns
import matplotlib.pyplot as plt

st_lit.title("📊 Bank Marketing Classification Models (UCI ML Dataset)")

# Provide GitHub download link for validation file
github_url = "https://github.com/2025aa05872/bank-marketing-streamlit/raw/main/data/bank_validation_small.csv"

try:
    response = requests.get(github_url)
    response.raise_for_status()  # check for errors
    csv_data = response.content

    st_lit.download_button(
        label="📥 Download Bank Marketing Validation Data File",
        data=csv_data,
        file_name="bank_validation_small.csv",
        mime="text/csv"
    )
except Exception as e:
    st_lit.error(f"Unable to fetch file from GitHub: {e}")

# Load scaler
scaler = None
if os.path.exists("model/saved_models/scaler.pkl"):
    with open("model/saved_models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

# Upload validation/test data FIRST
st_lit.subheader("Upload Validation/Test Data:")
uploaded_file = st_lit.file_uploader("Upload CSV with features (any delimiter):", type="csv")

# Model selection dropdown AFTER upload section
if os.path.exists("model/saved_models"):
    model_files = [f for f in os.listdir("model/saved_models") if f.endswith(".pkl") and f != "scaler.pkl"]
    display_names = ["Select a model"] + [name.replace("_", " ").replace(".pkl", "") for name in model_files]
    model_map = dict(zip(display_names[1:], model_files))  # skip "Select a model"

    model_choice = st_lit.selectbox("Choose a model:", display_names)

    # Only proceed if both file is uploaded and a model is selected
    if uploaded_file and model_choice != "Select a model":
        try:
            # Auto-detect delimiter
            user_data = pd.read_csv(uploaded_file, sep=None, engine="python")
        except Exception as e:
            st_lit.error(f"Error reading file: {e}")
            st_lit.stop()

        # Separate features and labels if available
        if "y" in user_data.columns:
            y_true = user_data["y"].map({"yes": 1, "no": 0}) if user_data["y"].dtype == "object" else user_data["y"]
            X_input = user_data.drop(columns=["y"])

            # One-hot encode categorical features
            X_input = pd.get_dummies(X_input, drop_first=True)

            # Apply scaling if scaler exists
            if scaler is not None:
                try:
                    X_input = scaler.transform(X_input)
                except Exception as e:
                    st_lit.error(f"Error applying scaler: {e}")

            # Load selected model
            model_file = model_map[model_choice]
            with open(f"model/saved_models/{model_file}", "rb") as f:
                model = pickle.load(f)

            # Predictions
            y_pred = model.predict(X_input)

            # Live evaluation metrics
            st_lit.subheader(f"Evaluation Metrics for {model_choice} (on Uploaded Data): ")
            metrics = {
                "Accuracy:": accuracy_score(y_true, y_pred),
                "Precision:": precision_score(y_true, y_pred, average="weighted"),
                "Recall:": recall_score(y_true, y_pred, average="weighted"),
                "F1:": f1_score(y_true, y_pred, average="weighted"),
                "MCC:": matthews_corrcoef(y_true, y_pred)
            }
            if hasattr(model, "predict_proba"):
                try:
                    y_prob = model.predict_proba(X_input)
                    metrics["AUC:"] = roc_auc_score(y_true, y_prob[:,1])
                except Exception:
                    metrics["AUC:"] = None
            else:
                metrics["AUC:"] = None

            st_lit.table(pd.DataFrame(metrics, index=["Score"]).T)

            # Confusion matrix
            st_lit.subheader("Confusion Matrix: ")
            cm = confusion_matrix(y_true, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            st_lit.pyplot(fig)

            # Classification report
            st_lit.subheader("Classification Report: ")
            report = classification_report(y_true, y_pred)
            st_lit.text(report)
        else:
            st_lit.warning("Uploaded file does not contain 'y' column (target). Metrics cannot be computed.")
    elif uploaded_file and model_choice == "Select a model":
        st_lit.warning("Please select a trained model from the dropdown to proceed.")
else:

    st_lit.warning("No saved models found. Please train models first.")

