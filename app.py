from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
MANIFEST = json.loads((MODEL_DIR / "model_manifest.json").read_text(encoding="utf-8"))

st.set_page_config(page_title="GRPR PET/CT Radiomics | RF", page_icon="🧬", layout="wide")

st.title("GRPR PET/CT Radiomics Prediction")
st.caption("Random forest multi-region radiomics platform | Research use only")
st.warning(
    "This tool provides research predictions and is not a standalone diagnostic system. "
    "Interpret results with the clinical and imaging context."
)

@st.cache_resource
def load_models():
    return {
        name: joblib.load(MODEL_DIR / info["model_file"])
        for name, info in MANIFEST["models"].items()
    }

models = load_models()

with st.sidebar:
    st.subheader("Model selection")
    selected_name = st.selectbox("Prediction model", list(MANIFEST["models"]))
    info = MANIFEST["models"][selected_name]
    st.metric("Selected features", len(info["features"]))
    st.metric("Decision threshold", f"{info['threshold']:.4f}")
    st.metric("Temporal AUC", f"{info['auc_temporal']:.3f}")
    st.caption("Threshold: development nested-OOF Youden index.")
    with st.expander("Selected feature list"):
        for feature in info["features"]:
            st.write(feature)

feature_list = MANIFEST["models"][selected_name]["features"]
threshold = float(MANIFEST["models"][selected_name]["threshold"])
model = models[selected_name]

manual_tab, batch_tab = st.tabs(["Manual prediction", "Batch CSV"])


def prepare_input(frame):
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Input must be a data table.")
    missing = [feature for feature in feature_list if feature not in frame.columns]
    if missing:
        raise ValueError("Missing required selected features: " + ", ".join(missing))
    selected = frame.loc[:, feature_list].copy()
    for feature in feature_list:
        selected[feature] = pd.to_numeric(selected[feature], errors="coerce")
    bad = [feature for feature in feature_list if selected[feature].isna().any() and feature not in MANIFEST["models"][selected_name]["imputer_statistics"]]
    if bad:
        raise ValueError("Missing or non-numeric values in: " + ", ".join(bad))
    medians = pd.Series(MANIFEST["models"][selected_name]["imputer_statistics"], dtype=float)
    selected = selected.fillna(medians)
    if "Multifocality" in selected and not selected["Multifocality"].isin([0, 1]).all():
        raise ValueError("Multifocality must be coded as 0 (no) or 1 (yes).")

    # Portable deployment models are pure RF classifiers; imputation is applied
    # from the locked training medians above, avoiding pickle-version coupling.
    return selected


def predict_frame(frame):
    full = prepare_input(frame)
    probability = model.predict_proba(full)[:, 1]
    result = frame.copy()
    result["PCa_probability"] = probability
    result["Predicted_class"] = np.where(probability >= threshold, "PCa", "Non-PCa")
    result["Decision_threshold"] = threshold
    return result


def show_results(result):
    st.dataframe(result, use_container_width=True, hide_index=True)
    if len(result) == 1:
        probability = float(result.iloc[0]["PCa_probability"])
        c1, c2, c3 = st.columns(3)
        c1.metric("Predicted class", result.iloc[0]["Predicted_class"])
        c2.metric("Estimated PCa probability", f"{probability:.1%}")
        c3.metric("Decision threshold", f"{threshold:.4f}")
        st.progress(min(max(probability, 0.0), 1.0))
    csv = result.to_csv(index=False).encode("utf-8-sig")
    st.download_button("Download predictions (CSV)", csv, "rf_predictions.csv", "text/csv")

with manual_tab:
    st.write("Enter the locked features for the selected RF model using the study feature definitions and units.")
    with st.form("manual_prediction"):
        columns = st.columns(3)
        values = {}
        for index, feature in enumerate(feature_list):
            with columns[index % 3]:
                if feature == "Multifocality":
                    values[feature] = st.selectbox(
                        feature, [0, 1],
                        format_func=lambda value: "No (0)" if value == 0 else "Yes (1)",
                    )
                else:
                    values[feature] = st.number_input(feature, value=0.0, format="%.6f")
        submitted = st.form_submit_button("Predict", type="primary", use_container_width=True)
    if submitted:
        try:
            show_results(predict_frame(pd.DataFrame([values])))
        except Exception as exc:
            st.error(str(exc))

with batch_tab:
    template = pd.DataFrame([{feature: (0 if feature == "Multifocality" else 0.0) for feature in feature_list}])
    st.download_button(
        "Download CSV template",
        template.to_csv(index=False).encode("utf-8-sig"),
        f"{selected_name.replace(' ', '_')}_RF_template.csv",
        "text/csv",
    )
    uploaded = st.file_uploader("Upload a CSV containing the selected feature columns", type=["csv"])
    if uploaded is not None:
        try:
            source = pd.read_csv(uploaded)
            st.caption(f"Loaded {len(source)} rows")
            st.dataframe(source.head(10), use_container_width=True, hide_index=True)
            if st.button("Run batch prediction", type="primary", disabled=source.empty):
                show_results(predict_frame(source))
        except Exception as exc:
            st.error(str(exc))

st.divider()
st.caption(
    f"Selected model: {selected_name}. The probability is the RF-estimated probability for PCa; "
    f"classification uses threshold {threshold:.4f}. Temporal validation AUC: {info['auc_temporal']:.3f}. "
    "This application does not replace biopsy or clinical judgement."
)
