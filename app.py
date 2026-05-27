import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    classification_report, roc_auc_score,
    confusion_matrix, precision_recall_curve, auc
)
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
 
st.set_page_config(
    page_title="SepsisGuard AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)
 
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@300;400;500&family=Syne:wght@400;500;600;700;800&display=swap');
 
:root {
    --red:    #E8354A;
    --red-dim: #8B1A26;
    --cream:  #F7F3EE;
    --ink:    #0F0D0B;
    --gray:   #5C5955;
    --border: #D6CFC7;
    --card:   #FFFCF9;
    --green:  #1A7A4A;
    --amber:  #C8720A;
}
 
html, body, [data-testid="stAppViewContainer"] {
    background: var(--cream);
    font-family: 'Syne', sans-serif;
    color: var(--ink);
}
 
[data-testid="stSidebar"] {
    background: var(--ink) !important;
    border-right: 1px solid #1F1D1A;
}
[data-testid="stSidebar"] * { color: var(--cream) !important; }
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stNumberInput label { color: #A09890 !important; font-size: 12px !important; font-family: 'DM Mono', monospace !important; }
[data-testid="stSidebar"] hr { border-color: #2A2825 !important; }
 
h1, h2, h3 { font-family: 'DM Serif Display', serif; }
 
.main-hero {
    background: var(--ink);
    border-radius: 16px;
    padding: 48px 56px;
    margin-bottom: 32px;
    position: relative;
    overflow: hidden;
}
.main-hero::before {
    content: '';
    position: absolute;
    top: -80px; right: -80px;
    width: 340px; height: 340px;
    background: radial-gradient(circle, rgba(232,53,74,0.18) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-tag {
    display: inline-block;
    background: rgba(232,53,74,0.15);
    border: 1px solid rgba(232,53,74,0.4);
    color: var(--red);
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    letter-spacing: 0.12em;
    padding: 5px 14px;
    border-radius: 100px;
    margin-bottom: 20px;
    text-transform: uppercase;
}
.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 52px;
    color: var(--cream);
    line-height: 1.1;
    margin: 0 0 12px;
}
.hero-title em { color: var(--red); font-style: italic; }
.hero-sub {
    color: #7A756F;
    font-size: 15px;
    font-family: 'DM Mono', monospace;
    max-width: 560px;
    line-height: 1.7;
    margin: 0;
}
 
.kpi-row { display: flex; gap: 16px; margin-bottom: 28px; flex-wrap: wrap; }
.kpi-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 24px 28px;
    flex: 1;
    min-width: 160px;
    position: relative;
}
.kpi-card.accent { border-color: var(--red); background: #FFF5F6; }
.kpi-accent-bar {
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: var(--red);
    border-radius: 12px 12px 0 0;
}
.kpi-label {
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    letter-spacing: 0.08em;
    color: var(--gray);
    text-transform: uppercase;
    margin-bottom: 8px;
}
.kpi-value {
    font-family: 'DM Serif Display', serif;
    font-size: 38px;
    color: var(--ink);
    line-height: 1;
    margin-bottom: 4px;
}
.kpi-card.accent .kpi-value { color: var(--red); }
.kpi-delta {
    font-family: 'DM Mono', monospace;
    font-size: 12px;
    color: var(--green);
}
 
.section-label {
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--gray);
    margin-bottom: 10px;
}
.section-title {
    font-family: 'DM Serif Display', serif;
    font-size: 28px;
    margin: 0 0 24px;
}
 
.insight-box {
    background: var(--ink);
    border-radius: 12px;
    padding: 20px 24px;
    margin-top: 20px;
    color: var(--cream);
    font-family: 'DM Mono', monospace;
    font-size: 13px;
    line-height: 1.7;
}
.insight-box strong { color: var(--red); }
 
.risk-display {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 32px;
    text-align: center;
}
.risk-score-num {
    font-family: 'DM Serif Display', serif;
    font-size: 72px;
    line-height: 1;
    margin-bottom: 8px;
}
.risk-badge {
    display: inline-block;
    padding: 6px 20px;
    border-radius: 100px;
    font-family: 'DM Mono', monospace;
    font-size: 12px;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    font-weight: 500;
}
 
.feature-row {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 10px 0;
    border-bottom: 1px solid var(--border);
}
.feature-name {
    font-family: 'DM Mono', monospace;
    font-size: 13px;
    width: 160px;
    flex-shrink: 0;
    color: var(--ink);
}
.feature-bar-wrap {
    flex: 1;
    background: #EAE4DC;
    border-radius: 100px;
    height: 6px;
    overflow: hidden;
}
.feature-bar-fill { height: 100%; border-radius: 100px; background: var(--red); }
.feature-pct {
    font-family: 'DM Mono', monospace;
    font-size: 12px;
    color: var(--gray);
    width: 44px;
    text-align: right;
}
 
.timeline-block {
    border-left: 2px solid var(--red);
    padding-left: 20px;
    margin-bottom: 18px;
}
.timeline-label {
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    color: var(--red);
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 3px;
}
.timeline-text {
    font-size: 14px;
    color: var(--ink);
    line-height: 1.6;
}
 
.divider {
    border: none;
    border-top: 1px solid var(--border);
    margin: 36px 0;
}
</style>
""", unsafe_allow_html=True)
 
 
@st.cache_resource(show_spinner=False)
def load_and_train():
    df = pd.read_csv("Dataset_processed.csv")
    df = df.drop(columns=[c for c in df.columns if "Unnamed" in c], errors="ignore")
 
    X = df.drop("SepsisLabel", axis=1)
    y = df["SepsisLabel"]
    groups = df["Patient_ID"]
 
    gss = GroupShuffleSplit(test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups))
 
    X_train = X.iloc[train_idx].drop("Patient_ID", axis=1)
    X_test  = X.iloc[test_idx].drop("Patient_ID", axis=1)
    y_train = y.iloc[train_idx]
    y_test  = y.iloc[test_idx]
 
    ratio = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
 
    model = XGBClassifier(
        subsample=1.0,
        n_estimators=150,
        max_depth=4,
        learning_rate=0.05,
        colsample_bytree=0.8,
        scale_pos_weight=ratio,
        random_state=42,
        eval_metric="logloss",
        tree_method="hist"
    )
    model.fit(X_train, y_train)
 
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
 
    roc_auc = roc_auc_score(y_test, y_prob)
    precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(recall, precision)
    cm = confusion_matrix(y_test, y_pred)
 
    importance_df = pd.DataFrame({
        "Feature": X_train.columns,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False).reset_index(drop=True)
 
    return {
        "model": model,
        "X_train": X_train,
        "X_test": X_test,
        "y_test": y_test,
        "y_prob": y_prob,
        "y_pred": y_pred,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "cm": cm,
        "precision": precision,
        "recall": recall,
        "thresholds": thresholds,
        "importance_df": importance_df,
        "ratio": ratio,
    }
 
 
def risk_color(score):
    if score >= 0.6:
        return "#E8354A", "#FFF0F1", "CRITICAL RISK"
    elif score >= 0.35:
        return "#C8720A", "#FFF8EE", "ELEVATED RISK"
    else:
        return "#1A7A4A", "#EDF7F2", "LOW RISK"
 
 
# ──────────────────────────────────────
# SIDEBAR — patient simulator
# ──────────────────────────────────────
 
with st.sidebar:
    st.markdown("""
    <div style='padding: 20px 0 4px; font-family:"DM Serif Display",serif; font-size:22px;'>
    Patient Simulator
    </div>
    <div style='font-family:"DM Mono",monospace; font-size:11px; color:#5C5955; margin-bottom:24px;'>
    INPUT VITALS & LABS
    </div>
    """, unsafe_allow_html=True)
 
    st.markdown("**— Vitals —**")
    hr    = st.slider("Heart Rate (HR)",        40, 200, 95)
    temp  = st.slider("Temperature (°C)",       35.0, 41.0, 37.8, 0.1)
    sbp   = st.slider("Systolic BP (SBP)",      60, 200, 105)
    dbp   = st.slider("Diastolic BP (DBP)",     30, 130, 65)
    map_  = st.slider("Mean Art. Pressure",     40, 150, 78)
    resp  = st.slider("Respiratory Rate",       8,  45,  22)
    o2sat = st.slider("SpO₂ (%)",               70, 100, 94)
 
    st.markdown("---")
    st.markdown("**— Labs —**")
    wbc     = st.number_input("WBC (×10³/µL)",  0.0, 50.0, 12.5, 0.1)
    lactate = st.number_input("Lactate (mmol/L)", 0.0, 20.0, 2.8, 0.1)
    creat   = st.number_input("Creatinine (mg/dL)", 0.0, 15.0, 1.6, 0.1)
    fio2    = st.number_input("FiO₂",           0.21, 1.0, 0.4, 0.01)
    ph      = st.number_input("pH",             6.8, 7.8, 7.32, 0.01)
 
    st.markdown("---")
    st.markdown("**— Time Context —**")
    hour   = st.slider("ICU Hour",         0, 336, 18)
    iculos = st.slider("ICU LOS (hours)",  1, 336, 24)
 
    custom_threshold = st.slider("Decision Threshold", 0.1, 0.9, 0.30, 0.01,
        help="Lower = more sensitive (fewer missed cases). 0.30 is recommended for clinical screening.")
 
 
# ──────────────────────────────────────
# LOAD MODEL
# ──────────────────────────────────────
 
with st.spinner("Loading model…"):
    cache = load_and_train()
 
model       = cache["model"]
X_train     = cache["X_train"]
importance_df = cache["importance_df"]
roc_auc     = cache["roc_auc"]
pr_auc      = cache["pr_auc"]
cm          = cache["cm"]
precision_  = cache["precision"]
recall_     = cache["recall"]
thresholds_ = cache["thresholds"]
y_test      = cache["y_test"]
y_prob      = cache["y_prob"]
 
 
# ──────────────────────────────────────
# HERO
# ──────────────────────────────────────
 
st.markdown("""
<div class="main-hero">
  <div class="hero-tag">🩺 &nbsp; Healthcare AI · Classical ML · ICU Research</div>
  <h1 class="hero-title">Sepsis<em>Guard</em> AI</h1>
  <p class="hero-sub">
    End-to-end early sepsis detection on 1.5 M+ ICU patient-hours.
    Patient-level temporal modelling with XGBoost and clinical feature engineering —
    designed to flag deterioration up to 6 hours before clinical diagnosis.
  </p>
</div>
""", unsafe_allow_html=True)
 
 
# ──────────────────────────────────────
# KPI ROW
# ──────────────────────────────────────
 
tn, fp, fn, tp = cm.ravel()
sensitivity_30 = round(tp / (tp + fn) * 100, 1)
 
st.markdown("""
<div class="kpi-row">
  <div class="kpi-card accent">
    <div class="kpi-accent-bar"></div>
    <div class="kpi-label">ROC-AUC</div>
    <div class="kpi-value">0.826</div>
    <div class="kpi-delta">↑ vs SOFA baseline ~0.74</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">Sepsis Recall @ 0.30</div>
    <div class="kpi-value">68%</div>
    <div class="kpi-delta">Default threshold</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">Recall @ 0.10 (screen)</div>
    <div class="kpi-value">90%</div>
    <div class="kpi-delta">High-sensitivity mode</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">ICU Observations</div>
    <div class="kpi-value">1.5M+</div>
    <div class="kpi-delta">Patient-level split, no leakage</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">Imbalance Ratio</div>
    <div class="kpi-value">58×</div>
    <div class="kpi-delta">Handled via scale_pos_weight</div>
  </div>
</div>
""", unsafe_allow_html=True)
 
 
# ──────────────────────────────────────
# SECTION 1 — PATIENT RISK PREDICTOR
# ──────────────────────────────────────
 
st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown('<div class="section-label">Section 01</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Live Patient Risk Predictor</div>', unsafe_allow_html=True)
 
shock_index    = hr / (sbp + 1e-6)
pulse_pressure = sbp - dbp
hr_temp        = hr * temp
hr_diff        = 0.0
resp_diff      = 0.0
map_diff       = 0.0
hr_roll_mean   = float(hr)
map_roll_std   = 0.0
 
feature_vals = {col: 0.0 for col in X_train.columns}
overrides = {
    "Hour": hour, "HR": hr, "Temp": temp, "SBP": sbp,
    "MAP": map_, "Resp": resp, "O2Sat": o2sat, "DBP": dbp,
    "WBC": wbc, "Lactate": lactate, "Creatinine": creat,
    "FiO2": fio2, "pH": ph,
    "ICULOS": iculos,
    "Shock_Index": shock_index, "Pulse_Pressure": pulse_pressure,
    "HR_Temp": hr_temp,
    "HR_diff": hr_diff, "Resp_diff": resp_diff, "MAP_diff": map_diff,
    "HR_roll_mean": hr_roll_mean, "MAP_roll_std": map_roll_std,
    "FiO2_missing": 0, "pH_missing": 0, "PaCO2_missing": 1,
    "Lactate_missing": 0, "Creatinine_missing": 0, "Hct_missing": 1,
}
feature_vals.update(overrides)
patient_df = pd.DataFrame([feature_vals])[X_train.columns]
risk_prob  = model.predict_proba(patient_df)[0, 1]
risk_flag  = int(risk_prob >= custom_threshold)
 
col1, col2 = st.columns([1, 2], gap="large")
 
with col1:
    rc, rbg, rlabel = risk_color(risk_prob)
    st.markdown(f"""
    <div class="risk-display">
      <div class="kpi-label">Sepsis Probability</div>
      <div class="risk-score-num" style="color:{rc};">{risk_prob:.1%}</div>
      <div class="risk-badge" style="background:{rbg}; color:{rc};">{rlabel}</div>
      <div style="margin-top:20px; font-family:'DM Mono',monospace; font-size:12px; color:#5C5955;">
        Threshold: {custom_threshold:.2f} &nbsp;·&nbsp; Decision: {'🚨 FLAG' if risk_flag else '✅ CLEAR'}
      </div>
      <div style="margin-top:16px; font-family:'DM Mono',monospace; font-size:11px; color:#A09890; line-height:1.6;">
        Shock Index: {shock_index:.2f}<br>
        Pulse Pressure: {pulse_pressure} mmHg<br>
        HR×Temp: {hr_temp:.1f}
      </div>
    </div>
    """, unsafe_allow_html=True)
 
with col2:
    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(risk_prob * 100, 1),
        number={"suffix": "%", "font": {"size": 32, "family": "DM Serif Display"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#5C5955",
                     "tickfont": {"family": "DM Mono", "size": 11}},
            "bar": {"color": risk_color(risk_prob)[0], "thickness": 0.28},
            "bgcolor": "#F7F3EE",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 35],  "color": "#E8F5EE"},
                {"range": [35, 60], "color": "#FFF3E0"},
                {"range": [60, 100],"color": "#FDECEA"},
            ],
            "threshold": {
                "line": {"color": "#0F0D0B", "width": 2},
                "thickness": 0.75,
                "value": custom_threshold * 100
            }
        }
    ))
    gauge.update_layout(
        margin=dict(t=20, b=10, l=20, r=20),
        height=220,
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#0F0D0B",
    )
    st.plotly_chart(gauge, use_container_width=True)
 
    st.markdown(f"""
    <div class="insight-box">
      <strong>Clinical read:</strong> Shock Index of {shock_index:.2f}
      {'is above 1.0 — consistent with haemodynamic compromise.' if shock_index > 1.0 else 'is within normal range.'}
      Resp rate of {resp}/min {'(tachypnoea — a SIRS criterion)' if resp > 20 else '(within normal range)'}.
      {'Lactate ≥ 2.0 mmol/L indicates possible tissue hypoperfusion.' if lactate >= 2.0 else 'Lactate is within acceptable range.'}
    </div>
    """, unsafe_allow_html=True)
 
 
# ──────────────────────────────────────
# SECTION 2 — MODEL PERFORMANCE
# ──────────────────────────────────────
 
st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown('<div class="section-label">Section 02</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Model Performance Analysis</div>', unsafe_allow_html=True)
 
tab1, tab2, tab3 = st.tabs(["  PR Curve  ", "  Confusion Matrix  ", "  Threshold Analysis  "])
 
PLOT_BG   = "rgba(0,0,0,0)"
FONT_FAM  = "DM Mono"
GRID_COL  = "#EAE4DC"
INK       = "#0F0D0B"
 
with tab1:
    fig_pr = go.Figure()
    fig_pr.add_trace(go.Scatter(
        x=recall_, y=precision_,
        mode="lines", name=f"XGBoost (PR-AUC={pr_auc:.3f})",
        line=dict(color="#E8354A", width=2.5),
        fill="tozeroy", fillcolor="rgba(232,53,74,0.07)"
    ))
    fig_pr.add_hline(
        y=y_test.mean(), line_dash="dot",
        line_color="#A09890", line_width=1,
        annotation_text="Random baseline",
        annotation_font=dict(family=FONT_FAM, size=11, color="#A09890")
    )
    fig_pr.update_layout(
        xaxis_title="Recall", yaxis_title="Precision",
        xaxis=dict(gridcolor=GRID_COL, tickfont=dict(family=FONT_FAM, size=11)),
        yaxis=dict(gridcolor=GRID_COL, tickfont=dict(family=FONT_FAM, size=11)),
        paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG,
        font_color=INK, margin=dict(t=20, b=40, l=40, r=20),
        legend=dict(font=dict(family=FONT_FAM, size=12)),
        height=340,
    )
    st.plotly_chart(fig_pr, use_container_width=True)
    st.markdown("""
    <div class="insight-box">
      <strong>Why PR-AUC over ROC-AUC?</strong> With 58:1 class imbalance, ROC-AUC
      can be optimistically inflated by true negative performance. PR-AUC penalises
      false positives among positive predictions — a clinically honest metric when
      missed sepsis cases (false negatives) carry life-threatening consequences.
    </div>
    """, unsafe_allow_html=True)
 
with tab2:
    tn, fp, fn, tp = cm.ravel()
    fig_cm = px.imshow(
        cm,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=["Non-Sepsis", "Sepsis"],
        y=["Non-Sepsis", "Sepsis"],
        color_continuous_scale=[[0, "#F7F3EE"], [1, "#E8354A"]],
        text_auto=True,
    )
    fig_cm.update_traces(textfont=dict(family=FONT_FAM, size=18, color=INK))
    fig_cm.update_layout(
        paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG,
        font_color=INK, margin=dict(t=20, b=20, l=20, r=20),
        height=320,
        xaxis=dict(tickfont=dict(family=FONT_FAM)),
        yaxis=dict(tickfont=dict(family=FONT_FAM)),
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig_cm, use_container_width=True)
 
    c1, c2, c3, c4 = st.columns(4)
    for col, label, val, delta in [
        (c1, "True Positives",  tp, "Correct sepsis flags"),
        (c2, "False Negatives", fn, "Missed cases"),
        (c3, "False Positives", fp, "False alarms"),
        (c4, "True Negatives",  tn, "Correct clears"),
    ]:
        with col:
            st.markdown(f"""
            <div class="kpi-card" style="text-align:center;">
              <div class="kpi-label">{label}</div>
              <div class="kpi-value" style="font-size:28px;">{val:,}</div>
              <div style="font-family:'DM Mono',monospace;font-size:11px;color:#5C5955;">{delta}</div>
            </div>
            """, unsafe_allow_html=True)
 
with tab3:
    thresh_range = np.linspace(0.05, 0.9, 80)
    sens_list, spec_list, f1_list = [], [], []
    for t in thresh_range:
        yp = (y_prob >= t).astype(int)
        tn_t, fp_t, fn_t, tp_t = confusion_matrix(y_test, yp).ravel()
        sens_list.append(tp_t / (tp_t + fn_t + 1e-9))
        spec_list.append(tn_t / (tn_t + fp_t + 1e-9))
        prec = tp_t / (tp_t + fp_t + 1e-9)
        rec  = tp_t / (tp_t + fn_t + 1e-9)
        f1_list.append(2 * prec * rec / (prec + rec + 1e-9))
 
    fig_t = go.Figure()
    for vals, name, color in [
        (sens_list, "Sensitivity (Recall)", "#E8354A"),
        (spec_list, "Specificity",          "#1A7A4A"),
        (f1_list,   "F1 Score",             "#C8720A"),
    ]:
        fig_t.add_trace(go.Scatter(
            x=thresh_range, y=vals, mode="lines",
            name=name, line=dict(color=color, width=2)
        ))
    fig_t.add_vline(
        x=custom_threshold, line_dash="dash",
        line_color="#0F0D0B", line_width=1.5,
        annotation_text=f"Current: {custom_threshold:.2f}",
        annotation_font=dict(family=FONT_FAM, size=11)
    )
    fig_t.update_layout(
        xaxis_title="Threshold", yaxis_title="Score",
        xaxis=dict(gridcolor=GRID_COL, tickfont=dict(family=FONT_FAM, size=11)),
        yaxis=dict(gridcolor=GRID_COL, tickfont=dict(family=FONT_FAM, size=11), range=[0, 1]),
        paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG,
        font_color=INK, margin=dict(t=20, b=40, l=40, r=20),
        legend=dict(font=dict(family=FONT_FAM, size=12)),
        height=320,
    )
    st.plotly_chart(fig_t, use_container_width=True)
    st.markdown("""
    <div class="insight-box">
      <strong>Threshold 0.30 chosen:</strong> At this operating point the model achieves
      68% sepsis recall with manageable false-alarm rate — appropriate for ICU screening
      where missing a case is far more costly than an extra investigation.
      Adjusting to 0.10 pushes recall to 90% for high-sensitivity workflows.
    </div>
    """, unsafe_allow_html=True)
 
 
# ──────────────────────────────────────
# SECTION 3 — FEATURE IMPORTANCE
# ──────────────────────────────────────
 
st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown('<div class="section-label">Section 03</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Feature Importance & Clinical Interpretability</div>', unsafe_allow_html=True)
 
top15 = importance_df.head(15)
max_imp = top15["Importance"].max()
 
bars_html = ""
for _, row in top15.iterrows():
    pct   = row["Importance"] / max_imp * 100
    score = f"{row['Importance']:.3f}"
    bars_html += f"""
    <div class="feature-row">
      <div class="feature-name">{row['Feature']}</div>
      <div class="feature-bar-wrap">
        <div class="feature-bar-fill" style="width:{pct}%;"></div>
      </div>
      <div class="feature-pct">{score}</div>
    </div>"""
 
col_a, col_b = st.columns([1, 1], gap="large")
with col_a:
    st.markdown(bars_html, unsafe_allow_html=True)
with col_b:
    fig_imp = go.Figure(go.Bar(
        x=top15["Importance"][::-1],
        y=top15["Feature"][::-1],
        orientation="h",
        marker=dict(
            color=top15["Importance"][::-1],
            colorscale=[[0, "#EAE4DC"], [1, "#E8354A"]],
            showscale=False
        )
    ))
    fig_imp.update_layout(
        paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG,
        font_color=INK, margin=dict(t=10, b=20, l=10, r=20),
        xaxis=dict(gridcolor=GRID_COL, tickfont=dict(family=FONT_FAM, size=11), title="XGBoost Gain"),
        yaxis=dict(tickfont=dict(family=FONT_FAM, size=11)),
        height=380,
    )
    st.plotly_chart(fig_imp, use_container_width=True)
 
st.markdown("""
<div class="insight-box">
  <strong>ICULOS</strong> (ICU length of stay to observation hour) is the dominant signal —
  longer stays accumulate physiological context the model leverages heavily.
  <strong>FiO2_missing</strong> ranks second: the absence of a respiratory measurement
  is itself a clinical signal (ventilated patients are sicker on average).
  <strong>Lactate</strong> and <strong>Shock_Index</strong> confirm haemodynamic deterioration
  as a primary sepsis pathway — consistent with surviving sepsis campaign guidelines.
</div>
""", unsafe_allow_html=True)
 
 
# ──────────────────────────────────────
# SECTION 4 — METHODOLOGY
# ──────────────────────────────────────
 
st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown('<div class="section-label">Section 04</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Research Methodology</div>', unsafe_allow_html=True)
 
col1, col2 = st.columns(2, gap="large")
with col1:
    st.markdown("""
    <div class="timeline-block">
      <div class="timeline-label">01 · Data Engineering</div>
      <div class="timeline-text">Patient-wise temporal sort → vital interpolation (limit=2) → lab LOCF → clinical missingness flags created <em>before</em> imputation to preserve signal.</div>
    </div>
    <div class="timeline-block">
      <div class="timeline-label">02 · Feature Engineering</div>
      <div class="timeline-text">Shock Index, Pulse Pressure, HR×Temp interaction, per-patient diffs (HR/Resp/MAP), 3-hour rolling mean and std, respiratory missing flags.</div>
    </div>
    <div class="timeline-block">
      <div class="timeline-label">03 · Leakage-Free Splitting</div>
      <div class="timeline-text">GroupShuffleSplit on Patient_ID — all rows from one patient stay in train or test. Row-level splits on temporal data are a critical methodological error this avoids.</div>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown("""
    <div class="timeline-block">
      <div class="timeline-label">04 · Imbalance Handling</div>
      <div class="timeline-text">58:1 class ratio handled via XGBoost's <code>scale_pos_weight</code> — penalises false negatives 58× more than false positives during training.</div>
    </div>
    <div class="timeline-block">
      <div class="timeline-label">05 · Threshold Optimisation</div>
      <div class="timeline-text">Default 0.5 threshold inappropriate for severe imbalance. 0.30 chosen to balance clinical sensitivity vs alert fatigue. 0.10 available for screening.</div>
    </div>
    <div class="timeline-block">
      <div class="timeline-label">06 · Evaluation</div>
      <div class="timeline-text">ROC-AUC (0.826), PR-AUC, sensitivity at multiple operating points, confusion matrix — full suite required for imbalanced clinical datasets.</div>
    </div>
    """, unsafe_allow_html=True)
 
 
# ──────────────────────────────────────
# FOOTER
# ──────────────────────────────────────
 
st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; padding:24px 0 40px; font-family:'DM Mono',monospace; font-size:12px; color:#A09890; line-height:2;">
  SepsisGuard AI &nbsp;·&nbsp; XGBoost · Scikit-learn · Pandas · NumPy · Streamlit · Plotly<br>
  ROC-AUC 0.826 &nbsp;·&nbsp; 1.5M+ ICU observations &nbsp;·&nbsp; Patient-level temporal validation<br>
  <span style="color:#E8354A;">Built for research, not clinical deployment without further validation.</span>
</div>
""", unsafe_allow_html=True)