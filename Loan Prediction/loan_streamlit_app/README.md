# Loan Status Prediction — Streamlit App

An interactive Streamlit app version of the Loan Status Prediction mini project
(originally a Jupyter/Colab notebook using Logistic Regression).

## What it does

- Trains a **Logistic Regression** model on the Loan Status Prediction dataset,
  using the same preprocessing as the original notebook:
  1. Forward-fill missing values (`ffill`)
  2. Label-encode categorical columns
  3. Standard-scale numeric features
  4. Fit `LogisticRegression`
- Lets you enter applicant details in a form and get a live **Approved / Not
  Approved** prediction with a confidence score.
- Shows model accuracy and a confusion matrix for the holdout test set.

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the app:**
   ```bash
   streamlit run app.py
   ```

   It will open in your browser at `http://localhost:8501`.

`Loan Status Prediction.csv` (the same 614-row dataset used in the original
notebook) is already included in this folder, so the app works out of the box.

## Notes on differences from the original notebook

- The `Loan_ID` column is dropped before training. It's just a unique
  identifier with no predictive value, and it isn't something a user would
  ever type into a prediction form — this is a small, standard cleanup, not
  a change to the model's actual logic.
- The model is retrained once per app session (cached with
  `st.cache_resource`) directly from your CSV, so it always reflects your
  exact dataset — there's no separately pickled model file to keep in sync.

## Deploying (optional)

To share this app with others (e.g. for your portfolio/resume link):

1. Push this folder to a public GitHub repo (include the CSV, or add a note
   in the README about where to get it if you'd rather not commit the raw
   data).
2. Go to [share.streamlit.io](https://share.streamlit.io), connect your
   GitHub repo, and deploy `app.py`. You'll get a public URL you can link to
   from your resume/portfolio.
