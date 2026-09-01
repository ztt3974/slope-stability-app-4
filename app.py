# -*- coding: utf-8 -*-
"""
Intelligent Slope Stability Prediction System (Streamlit Web App)
Based on a weighted ensemble model (XGBoost/LightGBM/CatBoost/RF/ET/GB)
Modules: Parameter Input -> Intelligent Prediction -> Result Display -> History
"""

import os
import csv
from datetime import datetime

import streamlit as st
import pandas as pd

import model_utils as mu

# ------------------------- Page config -------------------------
st.set_page_config(
    page_title="Intelligent Prediction System For Slope Stability",
    page_icon="⛰️",
    layout="wide",
)

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
HISTORY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "history", "predictions.csv")
HISTORY_COLUMNS = [
    "Time", "γ (kN/m³)", "C (kPa)", "φ (°)",
    "β (°)", "H (m)", "ru",
    "Result", "P(Stable)", "P(Unstable)",
]

# Sidebar parameter defs: (key, display name, unit, min, max, default, step, help)
PARAM_DEFS = [
    ("gamma", "Unit Weight γ", "kN/m³", 10.0, 30.0, 22.0, 0.1,
     "Unit weight of the geomaterial, typically 16–28"),
    ("cohesion", "Cohesion C", "kPa", 0.0, 150.0, 20.0, 0.5,
     "Soil cohesion, commonly 5–80 for cohesive soils"),
    ("friction", "Friction Angle φ", "°", 0.0, 55.0, 30.0, 0.5,
     "Internal friction angle of the geomaterial, typically 10–45"),
    ("slope_angle", "Slope Angle β", "°", 5.0, 90.0, 35.0, 0.5,
     "Angle between the slope face and the horizontal plane"),
    ("slope_height", "Slope Height H", "m", 1.0, 300.0, 30.0, 0.5,
     "Vertical height of the slope"),
    ("ru", "Pore Pressure Ratio ru", "", 0.0, 0.5, 0.25, 0.01,
     "Ratio of pore water pressure to total overburden stress, 0–0.5"),
]

DATA_SOURCE_EN = {
    "内置参考数据集": "Built-in reference dataset",
    "边坡稳定性数据（修正版3）.xlsx": "Slope stability dataset (Excel)",
}


# ------------------------- Helpers -------------------------
@st.cache_resource(show_spinner="Loading the prediction model...")
def get_model():
    """Load/train the ensemble model (globally cached)"""
    return mu.load_model()


def validate_params(params):
    """Validate parameter ranges; return a list of error messages"""
    errors = []
    checks = {
        "容重 Y(kg/m3)": (mu.PARAM_RANGES["容重 Y(kg/m3)"], "Unit Weight γ"),
        "粘聚力 C(kPa)": (mu.PARAM_RANGES["粘聚力 C(kPa)"], "Cohesion C"),
        "内摩擦角 φ(°)": (mu.PARAM_RANGES["内摩擦角 φ(°)"], "Friction Angle φ"),
        "坡角 β(°)": (mu.PARAM_RANGES["坡角 β(°)"], "Slope Angle β"),
        "坡高 H(m)": (mu.PARAM_RANGES["坡高 H(m)"], "Slope Height H"),
        "孔隙水压力比 r.": (mu.PARAM_RANGES["孔隙水压力比 r."], "Pore Pressure Ratio ru"),
    }
    for col, ((lo, hi), name) in checks.items():
        v = params[col]
        if v is None or (isinstance(v, float) and pd.isna(v)):
            errors.append(f"{name}: invalid input value")
        elif v < lo or v > hi:
            errors.append(f"{name}: {v} is out of the valid range [{lo}, {hi}]")
    return errors


def _adjust_value(key, delta, lo, hi):
    """Stepper button callback: increase/decrease within valid range"""
    v = st.session_state.get(key, lo) + delta
    st.session_state[key] = round(min(hi, max(lo, v)), 4)


# Legacy Chinese CSV column names -> current English columns
HISTORY_COLUMNS_LEGACY = {
    "预测时间": "Time",
    "容重γ(kN/m³)": "γ (kN/m³)",
    "粘聚力C(kPa)": "C (kPa)",
    "内摩擦角φ(°)": "φ (°)",
    "坡角β(°)": "β (°)",
    "坡高H(m)": "H (m)",
    "孔隙水压力比ru": "ru",
    "预测结果": "Result",
    "稳定概率": "P(Stable)",
    "不稳定概率": "P(Unstable)",
}


def load_history():
    """Load history records from CSV into session_state"""
    if "history" not in st.session_state:
        records = []
        if os.path.exists(HISTORY_FILE):
            try:
                df = pd.read_csv(HISTORY_FILE)
                df = df.rename(columns=HISTORY_COLUMNS_LEGACY)
                if "Result" in df.columns:
                    df["Result"] = df["Result"].map(
                        lambda x: "Stable" if str(x) in ("稳定", "Stable", "1")
                        else "Unstable"
                    )
                records = df.to_dict("records")
            except Exception:
                records = []
        st.session_state.history = records


def append_history(record):
    """Append one record and persist it to CSV"""
    st.session_state.history.append(record)
    os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
    df_new = pd.DataFrame([record], columns=HISTORY_COLUMNS)
    if os.path.exists(HISTORY_FILE):
        df_new.to_csv(HISTORY_FILE, mode="a", header=False, index=False,
                      encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)
    else:
        df_new.to_csv(HISTORY_FILE, index=False, encoding="utf-8-sig",
                      quoting=csv.QUOTE_MINIMAL)


# ------------------------- Header -------------------------
st.markdown(
    """
    <div style="padding:1.2rem 1.5rem;border-radius:0.8rem;
                background:linear-gradient(90deg,#1f3a5f 0%,#2e6da4 100%);
                color:white;">
        <h1 style="margin:0;font-size:1.8rem;">⛰️ Intelligent Prediction System For Slope Stability</h1>
        <p style="margin:0.4rem 0 0 0;opacity:0.9;">
            Based on Ensemble Model
            (XGBoost · LightGBM · CatBoost · Random Forest · Extra Trees · Gradient Boosting)
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.write("")

model, meta = get_model()
load_history()

# ========================= Module 1: Parameter Input =========================
with st.sidebar:
    st.header("📋 Slope Parameter Input")
    st.caption("Enter the six slope parameters (type a value directly, "
               "or fine-tune with the −/+ buttons)")

    inputs = {}
    for key, cname, unit, lo, hi, default, step, help_text in PARAM_DEFS:
        label = f"{cname} ({unit})" if unit else cname
        val_key = f"val_{key}"
        if val_key not in st.session_state:
            st.session_state[val_key] = float(default)
        fmt = "%.1f" if step >= 0.1 else "%.2f"

        st.caption(f"**{label}**")
        row = st.columns([0.7, 3.2, 0.7])
        with row[0]:
            st.button("−", key=f"dec_{key}", width="stretch",
                      on_click=_adjust_value, args=(val_key, -step, lo, hi))
        with row[1]:
            inputs[key] = st.number_input(
                label, min_value=float(lo), max_value=float(hi),
                step=step, format=fmt, key=val_key,
                label_visibility="collapsed", help=help_text,
            )
        with row[2]:
            st.button("+", key=f"inc_{key}", width="stretch",
                      on_click=_adjust_value, args=(val_key, step, lo, hi))
        st.write("")

    st.divider()
    predict_clicked = st.button(
        "🚀 Start Prediction", type="primary", width="stretch",
    )
    st.divider()
    with st.expander("ℹ️ Model Information"):
        source = meta.get("data_source", "—")
        st.write(f"**Data Source**: {DATA_SOURCE_EN.get(source, source)}")
        st.write(f"**Enhanced Features**: {meta.get('n_features', '—')} dimensions")
        st.write(f"**Validation Accuracy**: {meta.get('val_accuracy', 0):.2%}")
        st.write(f"**Decision Threshold**: {meta.get('threshold', 0.5):.3f}")
        weights = meta.get("weights", {})
        if weights:
            st.write("**Ensemble Model Weights**:")
            wdf = pd.DataFrame(
                sorted(weights.items(), key=lambda x: -x[1]),
                columns=["Base Model", "Weight"],
            )
            st.dataframe(wdf, hide_index=True, width="stretch")

# ========================= Module 2: Prediction + Module 3: Result Display =========================
params = {
    "容重 Y(kg/m3)": inputs["gamma"],
    "粘聚力 C(kPa)": inputs["cohesion"],
    "内摩擦角 φ(°)": inputs["friction"],
    "坡角 β(°)": inputs["slope_angle"],
    "坡高 H(m)": inputs["slope_height"],
    "孔隙水压力比 r.": inputs["ru"],
}

left, right = st.columns([3, 2])

with left:
    st.subheader("📥 Current Input Parameters")
    pnames = ["γ (kN/m³)", "C (kPa)", "φ (°)", "β (°)", "H (m)", "ru"]
    pvals = list(params.values())
    pshow = [f"{v:g}" for v in pvals]
    cols = st.columns(3)
    for i, (n, v) in enumerate(zip(pnames, pshow)):
        with cols[i % 3]:
            st.metric(n, v)
    st.info(
        "Click the **🚀 Start Prediction** button on the left. The system will "
        "assess slope stability using 45 physics-enhanced features and a "
        "six-model weighted ensemble."
    )

if predict_clicked:
    errors = validate_params(params)
    if errors:
        for e in errors:
            st.error(f"Parameter validation failed: {e}")
    else:
        result = mu.predict_single(model, meta, params)
        record = {
            "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "γ (kN/m³)": inputs["gamma"],
            "C (kPa)": inputs["cohesion"],
            "φ (°)": inputs["friction"],
            "β (°)": inputs["slope_angle"],
            "H (m)": inputs["slope_height"],
            "ru": inputs["ru"],
            "Result": "Stable" if result["label"] == 1 else "Unstable",
            "P(Stable)": round(result["proba_stable"], 4),
            "P(Unstable)": round(result["proba_unstable"], 4),
        }
        append_history(record)
        st.session_state.last_result = result
        st.rerun()

with right:
    st.subheader("📊 Prediction Result")
    result = st.session_state.get("last_result")
    if result is None:
        st.warning("No prediction yet. Enter the parameters on the left and "
                   "click 'Start Prediction'.")
    else:
        stable = result["label"] == 1
        p_stable = result["proba_stable"]
        p_unstable = result["proba_unstable"]

        if stable:
            st.markdown(
                f"""
                <div style="text-align:center;padding:1.5rem;border-radius:0.8rem;
                            background:linear-gradient(135deg,#e8f5e9,#c8e6c9);
                            border:2px solid #4caf50;">
                    <div style="font-size:2.2rem;">✅</div>
                    <div style="font-size:1.9rem;font-weight:700;color:#2e7d32;">
                        Slope Status: STABLE
                    </div>
                    <div style="color:#388e3c;margin-top:0.4rem;">
                        Probability of Stability {p_stable:.2%}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div style="text-align:center;padding:1.5rem;border-radius:0.8rem;
                            background:linear-gradient(135deg,#ffebee,#ffcdd2);
                            border:2px solid #f44336;">
                    <div style="font-size:2.2rem;">⚠️</div>
                    <div style="font-size:1.9rem;font-weight:700;color:#c62828;">
                        Slope Status: UNSTABLE
                    </div>
                    <div style="color:#d32f2f;margin-top:0.4rem;">
                        Probability of Instability {p_unstable:.2%}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.write("")
        st.progress(int(round(p_stable * 100)), text=f"P(Stable) {p_stable:.1%}")
        st.progress(int(round(p_unstable * 100)), text=f"P(Unstable) {p_unstable:.1%}")

        st.caption(
            f"Decision rule: P(Stable) = {p_stable:.4f} "
            f"{'≥' if stable else '<'} threshold {result['threshold']:.3f}"
        )

# ========================= Module 4: History =========================
st.divider()
st.subheader("🗂️ Prediction History")

history = st.session_state.get("history", [])
h_left, h_right = st.columns([3, 2])
with h_left:
    st.caption(f"{len(history)} record(s), auto-saved to local history/predictions.csv")
with h_right:
    if st.button("🗑️ Clear History"):
        st.session_state.history = []
        st.session_state.pop("last_result", None)
        if os.path.exists(HISTORY_FILE):
            os.remove(HISTORY_FILE)
        st.rerun()

if not history:
    st.info("No history yet. Once a prediction is completed, it will be saved here automatically.")
else:
    hdf = pd.DataFrame(history).reindex(columns=HISTORY_COLUMNS)
    hdf_display = hdf.copy()
    hdf_display["Result"] = hdf_display["Result"].map(
        lambda x: "✅ Stable" if str(x) in ("Stable", "稳定", "1") else "⚠️ Unstable"
    )
    st.dataframe(hdf_display, hide_index=True, width="stretch")

    csv_data = hdf.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
    st.download_button(
        "⬇️ Export History (CSV)",
        data=csv_data,
        file_name=f"slope_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        width="stretch",
    )

st.divider()
st.caption(
    "Disclaimer: The predictions of this system are for modeling research and "
    "preliminary assessment purposes only, and cannot replace detailed slope "
    "stability calculations and site investigations required by design codes."
)
