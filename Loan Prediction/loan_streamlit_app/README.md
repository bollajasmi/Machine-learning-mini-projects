# Loan Status Prediction — Streamlit Machine Learning App

An interactive machine learning application built with **Python and Streamlit** to predict whether a loan application is likely to be approved based on applicant information.

This project is an interactive version of the original Loan Status Prediction machine learning project developed using a Jupyter/Colab notebook.

## Project Overview

The application uses a **Logistic Regression** classification model trained on the Loan Status Prediction dataset.

Users can enter applicant details through the Streamlit interface and receive:

- Loan approval prediction
- Prediction confidence
- Model accuracy
- Confusion matrix

The model achieved **81.3% accuracy on the holdout test set**.

## Machine Learning Workflow

The application follows these steps:

1. Load the Loan Status Prediction dataset.
2. Remove the `Loan_ID` identifier column.
3. Handle missing values using forward-fill (`ffill`).
4. Encode categorical features using `LabelEncoder`.
5. Scale features using `StandardScaler`.
6. Split the dataset into training and testing sets.
7. Train a **Logistic Regression** classification model.
8. Evaluate the model using accuracy and a confusion matrix.
9. Use the trained model to make predictions from user-provided applicant details.

## Features

### Interactive Prediction

Users can enter applicant information such as:

- Gender
- Married status
- Dependents
- Education
- Self-employment status
- Applicant income
- Co-applicant income
- Loan amount
- Loan term
- Credit history
- Property area

The application then predicts the likely loan status.

### Prediction Confidence

The application displays a confidence score along with the prediction.

### Model Evaluation

The application displays:

- Holdout test accuracy
- Confusion matrix

## Technologies Used

- **Python**
- **Pandas** — data loading and preprocessing
- **NumPy** — numerical operations
- **Scikit-learn** — machine learning and preprocessing
- **Matplotlib** — confusion matrix visualization
- **Streamlit** — interactive web application

## Model

**Algorithm:** Logistic Regression

**Problem Type:** Binary Classification

**Dataset Size:** 614 records

**Evaluation:** Holdout test set

**Accuracy:** 81.3%

## Project Structure

```text
loan_streamlit_app/
│
├── app.py
├── Loan Status Prediction.csv
├── requirements.txt
└── README.md