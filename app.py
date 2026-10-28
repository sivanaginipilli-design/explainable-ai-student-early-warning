import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "model.joblib"
INFO_PATH = BASE_DIR / "model_info.json"

st.set_page_config(page_title="Student Early Warning System", page_icon="🎓", layout="wide")

st.title("🎓 Explainable AI Student Performance & Early Warning System")
st.caption("A decision-support prototype for identifying students who may need additional academic support.")

if not MODEL_PATH.exists() or not INFO_PATH.exists():
    st.error("Model files are missing. Run the Google Colab notebook, then download model.joblib and model_info.json into this project folder.")
    st.stop()

model = joblib.load(MODEL_PATH)
info = json.loads(INFO_PATH.read_text(encoding="utf-8"))
FEATURES = info["features"]
TARGET_NOTE = info["target_definition"]

st.info(
    "This is an educational prototype trained on a public student-performance dataset. "
    "It must not be used as the sole basis for decisions about students. Review predictions with the student and faculty."
)

with st.sidebar:
    st.header("Student academic information")
    st.write("Enter available early-term information.")
    age = st.slider("Age", min_value=15, max_value=22, value=17)
    studytime = st.selectbox(
        "Weekly study-time category",
        options=[1, 2, 3, 4],
        index=1,
        format_func=lambda x: {
            1: "1 — Less than 2 hours",
            2: "2 — 2 to 5 hours",
            3: "3 — 5 to 10 hours",
            4: "4 — More than 10 hours",
        }[x],
    )
    failures = st.slider("Previous class failures", min_value=0, max_value=4, value=0)
    absences = st.slider("Recorded absences", min_value=0, max_value=30, value=3)
    g1 = st.slider("First-period grade (0–20)", min_value=0, max_value=20, value=10)

    predict = st.button("Predict academic risk", type="primary", use_container_width=True)

st.subheader("How the system works")
c1, c2, c3 = st.columns(3)
c1.metric("Model", "XGBoost")
c2.metric("Explanation", "SHAP")
c3.metric("Output", "Risk estimate")

st.markdown(
    """
    The prototype uses early-term grade, recorded absences, study-time category, age, and previous failures.
    SHAP helps explain which input features pushed the prediction higher or lower.
    """
)

if predict:
    row = pd.DataFrame([{
        "age": age,
        "studytime": studytime,
        "failures": failures,
        "absences": absences,
        "G1": g1,
    }], columns=FEATURES)

    probability = float(model.predict_proba(row)[0, 1])
    label = "Higher risk" if probability >= 0.5 else "Lower risk"

    st.divider()
    st.subheader("Prediction")
    left, right = st.columns(2)
    left.metric("Estimated academic-risk probability", f"{probability:.1%}")
    right.metric("Model classification", label)
    st.progress(min(max(probability, 0.0), 1.0))

    if probability >= 0.5:
        st.warning("Suggested next step: arrange a supportive check-in with the student and review attendance, study habits, and subject-level needs.")
    else:
        st.success("No high-risk signal from this model for the entered values. Continue routine monitoring; a low score does not guarantee success.")

    st.subheader("Why did the model make this prediction?")
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(row)
        if isinstance(shap_values, list):
            values = np.asarray(shap_values[-1])[0]
        else:
            values = np.asarray(shap_values)
            if values.ndim == 3:
                values = values[0, :, -1]
            elif values.ndim == 2:
                values = values[0]

        explanation = pd.DataFrame({
            "Feature": FEATURES,
            "Entered value": [row.iloc[0][f] for f in FEATURES],
            "SHAP contribution": values,
        })
        explanation["Direction"] = np.where(
            explanation["SHAP contribution"] >= 0,
            "Pushes toward higher risk",
            "Pushes toward lower risk",
        )
        explanation = explanation.reindex(
            explanation["SHAP contribution"].abs().sort_values(ascending=False).index
        )
        st.dataframe(explanation.round(4), use_container_width=True, hide_index=True)

        fig, ax = plt.subplots(figsize=(8, 4))
        plot_data = explanation.sort_values("SHAP contribution")
        ax.barh(plot_data["Feature"], plot_data["SHAP contribution"])
        ax.axvline(0, linewidth=1)
        ax.set_xlabel("SHAP contribution (model output scale)")
        ax.set_title("Feature contributions for this prediction")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    except Exception as exc:
        st.error(f"Could not create the SHAP explanation: {exc}")

    st.subheader("Support plan (suggestions, not automated decisions)")
    suggestions = []
    if absences >= 5:
        suggestions.append("Discuss barriers to attendance and agree on a realistic attendance-support plan.")
    if g1 < 10:
        suggestions.append("Offer subject-specific remedial practice and a short follow-up assessment.")
    if studytime <= 1:
        suggestions.append("Help the student create a manageable weekly study timetable.")
    if failures > 0:
        suggestions.append("Arrange mentoring and review previously difficult topics.")
    if not suggestions:
        suggestions.append("Continue regular feedback and check progress after the next assessment.")
    for item in suggestions:
        st.write(f"• {item}")

st.divider()
st.caption(
    f"Target definition used during training: {TARGET_NOTE}. "
    "The risk threshold is a configurable prototype threshold, not a validated institutional policy."
)
