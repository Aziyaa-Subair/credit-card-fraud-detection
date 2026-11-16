import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

st.title("Credit Card Fraud Detector")


# ---------- Load saved objects (once, cached) ----------
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

@st.cache_data
def load_demo():
    return pd.read_csv("demo.csv")

@st.cache_data
def load_background():
    return pd.read_csv("background.csv")

@st.cache_resource
def load_scaler():
    return joblib.load("scaler.pkl")


model = load_model()
demo = load_demo()
background = load_background()
scaler = load_scaler()

# Threshold chosen on the validation set (see notebook)
THRESHOLD = 0.99


# ---------- Score one transaction ----------
idx = st.selectbox("Pick a transaction", demo.index)
row = demo.loc[[idx]]

X_row = row.drop(columns=["Class"])
prob = model.predict_proba(X_row)[:, 1][0]

# Demo data is scaled; reverse it to show the real amount
raw = scaler.inverse_transform(row[["Time", "Amount"]])

col1, col2 = st.columns(2)
col1.metric("Fraud score", f"{prob:.3f}")
col2.metric("Amount", f"{raw[0][1]:.2f}")

if prob >= THRESHOLD:
    st.error("🚩 Flagged as fraud")
else:
    st.success("✅ Approved")

st.write("Actually fraud?", "Yes" if row["Class"].iloc[0] == 1 else "No")


# ---------- Explain the score ----------
st.subheader("Why this score?")

explainer = shap.LinearExplainer(model, background)
sv = explainer(X_row)

shap.plots.waterfall(sv[0], show=False)
st.pyplot(plt.gcf())
plt.clf()


# ---------- Missed frauds across the demo set ----------
st.subheader("Frauds the model missed")

X_demo = demo.drop(columns=["Class"])
demo_probs = model.predict_proba(X_demo)[:, 1]

missed = demo[(demo["Class"] == 1) & (demo_probs < THRESHOLD)]
missed_raw = scaler.inverse_transform(missed[["Time", "Amount"]])
missed_view = missed[["Class"]].assign(
    amount=missed_raw[:, 1],
    score=demo_probs[missed.index],
)
st.write(len(missed), "frauds missed")
st.dataframe(missed_view)