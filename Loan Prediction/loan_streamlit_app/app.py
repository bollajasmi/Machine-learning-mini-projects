"""
Loan Status Prediction — Streamlit App
Built on top of Bolla Jasmi's Loan_Prediction.ipynb pipeline:
    ffill -> LabelEncoder (categoricals) -> StandardScaler -> LogisticRegression

To run:
    1. Place "Loan Status Prediction.csv" (same file used in the notebook) in this folder.
    2. pip install -r requirements.txt
    3. streamlit run app.py
"""

import os
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

st.set_page_config(page_title="Loan Status Prediction", page_icon="🏦", layout="centered")

DATA_PATH = "Loan Status Prediction.csv"

# ── Columns (same as the notebook, minus Loan_ID which is just an identifier) ──
CATEGORICAL_COLS = ["Gender", "Married", "Dependents", "Education", "Self_Employed", "Property_Area"]
NUMERIC_COLS = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term", "Credit_History"]
FEATURE_ORDER = ["Gender", "Married", "Dependents", "Education", "Self_Employed",
                  "ApplicantIncome", "CoapplicantIncome", "LoanAmount",
                  "Loan_Amount_Term", "Credit_History", "Property_Area"]


@st.cache_resource(show_spinner="Training model on your dataset...")
def train_model(csv_path: str):
    data = pd.read_csv(csv_path)

    # Drop the identifier column — it carries no predictive signal for new applicants
    if "Loan_ID" in data.columns:
        data = data.drop(columns=["Loan_ID"])

    # Same missing-value handling as the notebook
    data.ffill(inplace=True)

    # Same categorical encoding as the notebook (LabelEncoder per column),
    # but we keep each fitted encoder so the app can transform new user input consistently.
    encoders = {}
    for col in data.select_dtypes(include="object").columns:
        if col == "Loan_Status":
            continue
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        encoders[col] = le

    target_le = LabelEncoder()
    data["Loan_Status"] = target_le.fit_transform(data["Loan_Status"])  # N=0, Y=1

    X = data[FEATURE_ORDER]
    y = data["Loan_Status"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = X_train.fillna(X_train.mean())
    X_test = X_test.fillna(X_train.mean())

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=2000)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    return model, scaler, encoders, target_le, acc, cm, len(data)


# ── Sidebar: dataset status ──────────────────────────────────────────────────
st.sidebar.header("About")
st.sidebar.write(
    "This app trains a Logistic Regression model on the Loan Status Prediction "
    "dataset (same pipeline as the original notebook: fill missing values, "
    "label-encode categoricals, scale features, then fit Logistic Regression)."
)

if not os.path.exists(DATA_PATH):
    st.error(
        f"Dataset file not found: **{DATA_PATH}**\n\n"
        "Place your `Loan Status Prediction.csv` (the same file used in the notebook) "
        "in this app's folder, then rerun the app."
    )
    st.stop()

model, scaler, encoders, target_le, acc, cm, n_rows = train_model(DATA_PATH)

st.sidebar.metric("Model accuracy (holdout set)", f"{acc * 100:.1f}%")
st.sidebar.caption(f"Trained on {n_rows} records.")

# ── Main UI ───────────────────────────────────────────────────────────────────
st.title("🏦 Loan Status Prediction")
st.write(
    "Fill in the applicant details below to predict whether the loan is likely "
    "to be **approved** or **not approved**."
)

with st.form("loan_form"):
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        married = st.selectbox("Married", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
        education = st.selectbox("Education", ["Graduate", "Not Graduate"])
        self_employed = st.selectbox("Self Employed", ["No", "Yes"])
        property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

    with col2:
        applicant_income = st.number_input("Applicant Income (monthly)", min_value=0, value=5000, step=100)
        coapplicant_income = st.number_input("Coapplicant Income (monthly)", min_value=0, value=0, step=100)
        loan_amount = st.number_input("Loan Amount (in thousands)", min_value=0, value=120, step=5)
        loan_term = st.selectbox(
            "Loan Amount Term (days)",
            [360, 180, 480, 300, 240, 120, 60, 84, 36, 12],
            index=0,
        )
        credit_history = st.selectbox(
            "Credit History",
            options=[1.0, 0.0],
            format_func=lambda v: "Has repaid past debts (1)" if v == 1.0 else "No / poor credit record (0)",
        )

    submitted = st.form_submit_button("Predict Loan Status", use_container_width=True)

if submitted:
    row = {
        "Gender": gender,
        "Married": married,
        "Dependents": dependents,
        "Education": education,
        "Self_Employed": self_employed,
        "ApplicantIncome": applicant_income,
        "CoapplicantIncome": coapplicant_income,
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": loan_term,
        "Credit_History": credit_history,
        "Property_Area": property_area,
    }
    input_df = pd.DataFrame([row])[FEATURE_ORDER]

    # Encode categoricals using the SAME encoders fit during training
    for col in CATEGORICAL_COLS:
        le = encoders[col]
        val = input_df.at[0, col]
        if val not in le.classes_:
            st.error(f"Unrecognized value '{val}' for {col}. Cannot encode.")
            st.stop()
        input_df[col] = le.transform([val])

    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
    proba = model.predict_proba(input_scaled)[0]

    result_label = target_le.inverse_transform([prediction])[0]  # 'Y' or 'N'

    st.divider()
    if result_label == "Y":
        st.success(f"✅ Loan likely **Approved** (confidence: {max(proba) * 100:.1f}%)")
    else:
        st.error(f"❌ Loan likely **Not Approved** (confidence: {max(proba) * 100:.1f}%)")

    st.caption(
        "This prediction is based on a model trained on historical data and is for "
        "demonstration purposes only — it is not a real lending decision."
    )

# ── Model performance section ─────────────────────────────────────────────────
with st.expander("📊 View model performance on the holdout test set"):
    st.write(f"**Accuracy:** {acc * 100:.2f}%")
    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Approved", "Approved"])
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    st.pyplot(fig)
