import streamlit as st
import pandas as pd
import joblib
import os

st.title("🩺 Disease Prediction (Breast Cancer)")
st.write("This demo uses sklearn's built-in Breast Cancer dataset.")
st.write("👉 To train the model manually, click 'Train model now' or run `src/train.py` once.")

# ---- Train model ----
if st.button("Train model now (quick)"):
    import subprocess, sys
    try:
        subprocess.run([sys.executable, "src/train.py"], check=True)
        st.success("✅ Training completed successfully.")
    except Exception as e:
        st.error(f"Training failed: {e}")

# ---- Load model ----
if st.button("Load model"):
    model_path = "models/disease_rf.joblib"
    if os.path.exists(model_path):
        model_pack = joblib.load(model_path)
        st.session_state["model_pack"] = model_pack
        st.success("✅ Model loaded successfully.")
    else:
        st.error("❌ Model file not found. Please train it first.")

# ---- Feature importance ----
if st.button("Show feature importances"):
    if "model_pack" in st.session_state:
        import matplotlib.pyplot as plt
        from sklearn.datasets import load_breast_cancer
        model = st.session_state["model_pack"]["model"]
        try:
            importances = model.feature_importances_
            feat_names = load_breast_cancer(as_frame=True).feature_names
            df = pd.DataFrame({
                "Feature": feat_names,
                "Importance": importances / importances.max()
            }).sort_values("Importance", ascending=False).head(10)
            st.table(df)
        except Exception:
            st.warning("⚠️ This model does not have feature_importances_.")
    else:
        st.warning("⚠️ Please load the model first.")

# ---- File upload and prediction ----
st.write("Upload a CSV with feature columns (or use default structure).")
uploaded = st.file_uploader("CSV file (one sample row)", type=["csv"])

if uploaded is not None:
    if "model_pack" in st.session_state:
        df = pd.read_csv(uploaded)
        model = st.session_state["model_pack"]["model"]
        scaler = st.session_state["model_pack"]["scaler"]

        X = scaler.transform(df.values)
        pred = model.predict(X)
        prob = model.predict_proba(X)[:, 1]

        st.success(f"✅ Prediction: {int(pred[0])} | Probability of positive: {prob[0]:.4f}")
    else:
        st.warning("⚠️ Load or train the model before making predictions.") 