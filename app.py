import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report
)

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

st.title("💳 Credit Card Fraud Detection")
st.write("Machine Learning Based Fraud Detection System")

# Upload dataset
uploaded_file = st.file_uploader(
    "Upload Credit Card Dataset",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    if "Class" not in df.columns:
        st.error("Dataset must contain a 'Class' column.")
        st.stop()

    st.subheader("Dataset Information")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Transactions", len(df))
    col2.metric("Normal Transactions", int((df["Class"] == 0).sum()))
    col3.metric("Fraud Transactions", int((df["Class"] == 1).sum()))

    # Data visualization
    st.subheader("Transaction Distribution")

    counts = df["Class"].value_counts().sort_index()

    fig, ax = plt.subplots()
    ax.bar(
        ["Normal", "Fraud"],
        [counts.get(0, 0), counts.get(1, 0)]
    )
    ax.set_ylabel("Number of Transactions")
    st.pyplot(fig)

    # Prepare data
    X = df.drop("Class", axis=1)
    y = df["Class"]

    X = X.select_dtypes(include=[np.number])
    X = X.replace([np.inf, -np.inf], np.nan)

    valid_rows = X.notna().all(axis=1)
    X = X.loc[valid_rows]
    y = y.loc[valid_rows]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train model
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    )

    with st.spinner("Training machine learning model..."):
        model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    # Performance
    st.subheader("Model Performance")

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, zero_division=0
    )
    recall = recall_score(
        y_test, predictions, zero_division=0
    )

    c1, c2, c3 = st.columns(3)

    c1.metric("Accuracy", f"{accuracy * 100:.2f}%")
    c2.metric("Precision", f"{precision * 100:.2f}%")
    c3.metric("Recall", f"{recall * 100:.2f}%")

    # Confusion matrix
    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y_test, predictions)

    fig, ax = plt.subplots()
    ax.imshow(cm, cmap="Blues")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1], ["Normal", "Fraud"])
    ax.set_yticks([0, 1], ["Normal", "Fraud"])

    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center")

    st.pyplot(fig)

    st.subheader("Classification Report")
    report = classification_report(
        y_test,
        predictions,
        output_dict=True,
        zero_division=0
    )
    st.dataframe(pd.DataFrame(report).transpose())

    # Predict uploaded transactions
    st.subheader("Test New Transactions")

    new_file = st.file_uploader(
        "Upload transactions without the Class column",
        type=["csv"],
        key="new_transactions"
    )

    if new_file is not None:
        new_df = pd.read_csv(new_file)

        if set(X.columns).issubset(new_df.columns):
            new_X = new_df[X.columns].replace(
                [np.inf, -np.inf], np.nan
            )

            if new_X.isna().any().any():
                st.error("Please remove missing or invalid values.")
            else:
                new_scaled = scaler.transform(new_X)
                new_predictions = model.predict(new_scaled)

                result = new_df.copy()
                result["Prediction"] = [
                    "Fraud" if p == 1 else "Legitimate"
                    for p in new_predictions
                ]

                st.dataframe(result)

                st.download_button(
                    "Download Predictions",
                    result.to_csv(index=False),
                    "fraud_predictions.csv",
                    "text/csv"
                )
        else:
            st.error("Uploaded file does not have the required columns.")

else:
    st.info("Please upload your creditcard.csv dataset to begin.")