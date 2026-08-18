"""
FaceAge AI — Streamlit Dashboard
Rich dark-mode UI matching the original Flask dashboard design.
"""

import os, sys, time, io, base64
import numpy as np
from PIL import Image
import streamlit as st
import pandas as pd

# ── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FaceAge AI — Facial Age & Gender Estimation",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Global CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

html, body, .stApp { background:#070b14; color:#f8fafc; font-family:'Plus Jakarta Sans',sans-serif; }

/* hide default Streamlit header/footer */
#MainMenu, footer, header { visibility:hidden; }

/* Remove extra top padding */
.block-container { padding-top: 1rem !important; }

/* Cards */
.card {
  background: linear-gradient(135deg,#0d1424 0%,#0a1020 100%);
  border: 1px solid rgba(56,189,248,.15);
  border-radius: 16px;
  padding: 20px 22px;
  margin-bottom: 12px;
}
.card-label {
  font-size:.72rem; font-weight:700; letter-spacing:1.5px;
  text-transform:uppercase; color:#64748b; margin-bottom:6px;
}

/* Age hero */
.age-hero {
  background: linear-gradient(135deg,#0d1424 0%,#0a1020 100%);
  border: 1px solid rgba(56,189,248,.2);
  border-radius: 18px;
  padding: 28px 22px;
  text-align: center;
}
.age-number {
  font-size:5.2rem; font-weight:800; line-height:1;
  color:#fff; letter-spacing:-3px; margin:8px 0 4px;
  text-shadow: 0 0 40px rgba(56,189,248,.25);
}
.age-dash { font-size:.8rem; font-weight:700; color:#475569; letter-spacing:3px; margin-bottom:14px; }
.age-badge {
  display:inline-block; padding:6px 18px;
  background:rgba(34,197,94,.1); border:1.5px solid rgba(34,197,94,.4);
  color:#22c55e; border-radius:100px; font-weight:700; font-size:.9rem; margin-bottom:14px;
}
.age-range { font-size:1.5rem; font-weight:800; color:#38bdf8; margin-top:4px; }
.age-range-label { font-size:.72rem; color:#64748b; }

/* Stat pill cards */
.stat-card {
  background:#080e1c; border:1px solid rgba(56,189,248,.1);
  border-radius:14px; padding:16px 18px; margin-bottom:10px;
  border-left-width: 3px; border-left-style: solid;
}
.stat-val { font-size:1.5rem; font-weight:800; margin:4px 0 2px; }
.stat-sub { font-size:.68rem; font-weight:600; }

/* Upload area */
.upload-zone {
  background:#080e1c; border:2px dashed rgba(56,189,248,.25);
  border-radius:14px; padding:20px; text-align:center; color:#64748b;
}

/* Section headers */
.section-header {
  font-size:.72rem; font-weight:700; letter-spacing:1.5px;
  text-transform:uppercase; color:#38bdf8;
  border-bottom:1px solid rgba(56,189,248,.15);
  padding-bottom:8px; margin-bottom:14px;
}

/* Status pills */
.pill-green { display:inline-block; padding:4px 12px; border-radius:100px; font-size:.72rem; font-weight:700; background:rgba(34,197,94,.1); border:1px solid rgba(34,197,94,.3); color:#22c55e; margin-right:6px; }
.pill-blue  { display:inline-block; padding:4px 12px; border-radius:100px; font-size:.72rem; font-weight:700; background:rgba(56,189,248,.1); border:1px solid rgba(56,189,248,.3); color:#38bdf8; margin-right:6px; }
.pill-red   { display:inline-block; padding:4px 12px; border-radius:100px; font-size:.72rem; font-weight:700; background:rgba(239,68,68,.1); border:1px solid rgba(239,68,68,.3); color:#ef4444; margin-right:6px; }

/* Error box */
.error-box {
  background:rgba(239,68,68,.07); border:1px solid rgba(239,68,68,.3);
  border-radius:14px; padding:28px; text-align:center;
}

/* KPI bar */
.kpi-bar {
  background:#0d1424; border:1px solid rgba(56,189,248,.12);
  border-radius:14px; padding:18px 24px;
  display:flex; justify-content:space-around; flex-wrap:wrap; gap:12px;
}
.kpi-item { text-align:center; }
.kpi-val { font-size:1.6rem; font-weight:800; color:#38bdf8; }
.kpi-label { font-size:.68rem; color:#64748b; font-weight:600; }

/* Heatmap pair */
.cam-pair { display:flex; gap:12px; }
.cam-box { flex:1; background:#080e1c; border:1px solid rgba(56,189,248,.1); border-radius:12px; padding:10px; text-align:center; }
.cam-label { font-size:.68rem; color:#64748b; font-weight:600; margin-bottom:6px; }
</style>
""", unsafe_allow_html=True)

# ── Model loader (cached) ──────────────────────────────────────────────────
@st.cache_resource(show_spinner="🧠 Loading ConvNeXt-Small + Gender ViT...")
def get_predictor():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from inference import AgePredictor
    base = os.path.dirname(os.path.abspath(__file__))
    return AgePredictor(
        model_path=os.path.join(base, "models", "best_age_model.pt"),
        metrics_path=os.path.join(base, "results", "final_metrics.json"),
    )

def img_to_b64(pil_img, fmt="JPEG", quality=88):
    buf = io.BytesIO()
    pil_img.save(buf, format=fmt, quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

# ══════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style="text-align:center;padding:10px 0 18px;">
  <span style="font-size:2rem;">🧠</span>
  <span style="font-size:1.9rem;font-weight:800;color:#fff;"> FaceAge </span>
  <span style="font-size:1.9rem;font-weight:800;color:#38bdf8;">AI</span>
  <div style="font-size:.82rem;color:#64748b;margin-top:2px;">
    AI-Powered Multi-Attribute Facial Estimation &nbsp;·&nbsp; ConvNeXt-Small + DLDL
  </div>
  <div style="margin-top:10px;">
    <span class="pill-green">● Model Loaded</span>
    <span class="pill-blue">ConvNeXt-Small (DLDL)</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════
if "result" not in st.session_state:
    st.session_state.result = None
if "img_display" not in st.session_state:
    st.session_state.img_display = None
if "elapsed_ms" not in st.session_state:
    st.session_state.elapsed_ms = None

# ══════════════════════════════════════════════════════════════════════════
# MAIN 3-COLUMN LAYOUT
# ══════════════════════════════════════════════════════════════════════════
col_left, col_mid, col_right = st.columns([1, 1.3, 1], gap="medium")

# ── LEFT: Upload ───────────────────────────────────────────────────────────
with col_left:
    st.markdown('<div class="section-header">📸 SELECT PORTRAIT PHOTO</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("", type=["jpg","jpeg","png","webp"], label_visibility="collapsed")

    if uploaded:
        pil = Image.open(uploaded).convert("RGB")
        st.session_state.img_display = pil
        st.image(pil, use_container_width=True, caption="Uploaded Portrait")

        if st.button("⚡ Analyze Image", use_container_width=True, type="primary"):
            predictor = get_predictor()
            arr = np.array(pil)
            with st.spinner("Analyzing (MTCNN gate → ConvNeXt + TTA)…"):
                t0 = time.perf_counter()
                res = predictor.predict(arr)
                st.session_state.elapsed_ms = int((time.perf_counter() - t0) * 1000)
            st.session_state.result = res
            st.rerun()
    else:
        st.markdown("""
        <div class="upload-zone">
          <div style="font-size:2rem;">📷</div>
          <div style="margin-top:8px;font-size:.85rem;">Click above to browse or drop a photo</div>
          <div style="font-size:.72rem;margin-top:4px;">JPG · PNG · WEBP</div>
        </div>
        """, unsafe_allow_html=True)
        st.session_state.result = None

# ── MID + RIGHT: Results ───────────────────────────────────────────────────
result = st.session_state.result
elapsed_ms = st.session_state.elapsed_ms

with col_mid:
    st.markdown('<div class="section-header">🎯 PREDICTED AGE</div>', unsafe_allow_html=True)

    if result is None:
        # Blank initial state
        st.markdown("""
        <div class="age-hero" style="padding:48px 22px;">
          <div style="font-size:3rem;opacity:.15;margin-bottom:16px;">👤</div>
          <div style="color:#475569;font-size:.9rem;">Upload a photo and click<br><strong style="color:#38bdf8;">⚡ Analyze Image</strong> to begin</div>
        </div>
        """, unsafe_allow_html=True)

    elif "error" in result:
        st.markdown(f"""
        <div class="error-box">
          <div style="font-size:2.5rem;margin-bottom:12px;">⚠️</div>
          <div style="font-size:1.1rem;font-weight:700;color:#ef4444;margin-bottom:8px;">Validation Rejected</div>
          <div style="font-size:.85rem;color:#94a3b8;">{result['error']}</div>
        </div>
        <div style="margin-top:10px;">
          <span class="pill-red">✕ Verification Failed</span>
          <span class="pill-red">✕ Unusable Photo</span>
        </div>
        """, unsafe_allow_html=True)

    else:
        age_val   = result.get("predicted_age_int", int(round(result["predicted_age"])))
        group_val = result["predicted_age_group"]
        range_val = result["likely_age_range"]

        st.markdown(f"""
        <div class="age-hero">
          <div class="card-label">PREDICTED AGE</div>
          <div class="age-number">{age_val}</div>
          <div class="age-dash">— YEARS —</div>
          <div class="age-badge">👤 {group_val}</div>
          <div class="age-range-label">Estimated Age Range (95% CI)</div>
          <div class="age-range">{range_val} Years</div>
        </div>
        <div style="margin-top:10px;">
          <span class="pill-green">✓ Verified</span>
          <span class="pill-green">✓ Face Detected</span>
        </div>
        """, unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="section-header">📊 ATTRIBUTES & METRICS</div>', unsafe_allow_html=True)

    if result is None:
        for label, icon, color in [("Predicted Gender","♀","#f472b6"), ("Age Confidence","🛡","#22c55e"), ("Inference Time","⏱","#38bdf8"), ("Age Cohort","🎂","#a78bfa")]:
            st.markdown(f"""
            <div class="stat-card" style="border-left-color:{color}">
              <div class="card-label">{icon} {label}</div>
              <div class="stat-val" style="color:{color}">—</div>
            </div>
            """, unsafe_allow_html=True)

    elif "error" in result:
        st.markdown("""
        <div class="stat-card" style="border-left-color:#ef4444">
          <div class="card-label">STATUS</div>
          <div class="stat-val" style="color:#ef4444">N/A</div>
          <div class="stat-sub" style="color:#ef4444">Rejected</div>
        </div>
        """, unsafe_allow_html=True)

    else:
        gender      = result.get("predicted_gender", "Unknown")
        gender_conf = result.get("gender_confidence", "—")
        conf_pct    = result.get("confidence_pct", 0)
        cohort      = result.get("predicted_age_group", "—")
        bracket     = result.get("likely_age_range", "—")
        is_female   = gender.lower() == "female"
        g_color     = "#f472b6" if is_female else "#38bdf8"
        g_icon      = "👩" if is_female else "👨"

        st.markdown(f"""
        <div class="stat-card" style="border-left-color:{g_color}">
          <div class="card-label">{g_icon} Predicted Gender</div>
          <div class="stat-val" style="color:{g_color}">{gender}</div>
          <div class="stat-sub" style="color:{g_color}">Confidence: {gender_conf}</div>
        </div>
        <div class="stat-card" style="border-left-color:#22c55e">
          <div class="card-label">🛡 Age Confidence</div>
          <div class="stat-val" style="color:#22c55e">{conf_pct}%</div>
          <div class="stat-sub" style="color:#22c55e">Intrinsic ±1.96σ Metric</div>
        </div>
        <div class="stat-card" style="border-left-color:#38bdf8">
          <div class="card-label">⏱ Inference Time</div>
          <div class="stat-val" style="color:#fff">{elapsed_ms} ms</div>
          <div class="stat-sub" style="color:#22c55e">● TTA Enabled</div>
        </div>
        <div class="stat-card" style="border-left-color:#a78bfa">
          <div class="card-label">🎂 Age Cohort</div>
          <div class="stat-val" style="color:#a78bfa">{cohort}</div>
          <div class="stat-sub" style="color:#64748b">Bracket: {bracket} yrs</div>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# BOTTOM: Grad-CAM + Distribution Chart
# ══════════════════════════════════════════════════════════════════════════
st.markdown("---")

if result and "error" not in result:
    cam_col, dist_col = st.columns([1, 1.3], gap="medium")

    with cam_col:
        st.markdown('<div class="section-header">🧠 AI EXPLANATION (GRAD-CAM HEATMAP)</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div style="font-size:.72rem;color:#64748b;font-weight:600;margin-bottom:4px;">Original Face Crop</div>', unsafe_allow_html=True)
            if result.get("original_face_b64"):
                st.image(result["original_face_b64"], use_container_width=True)
            else:
                st.markdown('<div style="color:#475569;text-align:center;font-size:.8rem;padding:30px 0;">—</div>', unsafe_allow_html=True)
        with c2:
            st.markdown('<div style="font-size:.72rem;color:#64748b;font-weight:600;margin-bottom:4px;">AI Attention Focus</div>', unsafe_allow_html=True)
            if result.get("gradcam_b64"):
                st.image(result["gradcam_b64"], use_container_width=True)
            else:
                st.markdown('<div style="color:#475569;text-align:center;font-size:.8rem;padding:30px 0;">—</div>', unsafe_allow_html=True)

    with dist_col:
        st.markdown('<div class="section-header">📊 AGE PREDICTION DISTRIBUTION (DLDL PROBABILITY)</div>', unsafe_allow_html=True)
        if result.get("distribution_bins"):
            df = pd.DataFrame(result["distribution_bins"])
            df["Probability (%)"] = (df["probability"] * 100).round(1)
            st.bar_chart(df.set_index("bin")["Probability (%)"],
                         color="#38bdf8", use_container_width=True, height=220)
        else:
            st.markdown('<div style="color:#475569;text-align:center;padding:40px 0;">Distribution unavailable</div>', unsafe_allow_html=True)

elif result and "error" in result:
    st.markdown("""
    <div style="text-align:center;padding:24px;color:#475569;font-size:.85rem;">
      ⚠️ Distribution unavailable due to validation rejection
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# KPI SUMMARY BAR
# ══════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown('<div class="section-header">🏆 MODEL PERFORMANCE BENCHMARK SUMMARY</div>', unsafe_allow_html=True)

kpis = [
    ("MAE","5.6 yrs","Mean Absolute Error on 66k validation set"),
    ("RMSE","7.9 yrs","Root Mean Squared Error"),
    ("±5y Accuracy","81.4%","Predictions within 5 years"),
    ("±10y Accuracy","94.2%","Cumulative within 10 years"),
    ("Train Images","309,462","Verified training images"),
    ("Val Images","66,313","Validation benchmark set"),
]
cols = st.columns(len(kpis))
for col, (label, val, tip) in zip(cols, kpis):
    col.metric(label=label, value=val, help=tip)

# Footer
st.markdown("""
<div style="text-align:center;color:#334155;font-size:.72rem;margin-top:24px;padding-bottom:10px;">
  FaceAge AI &nbsp;·&nbsp; Deep Learning Facial Age & Gender Estimation &nbsp;·&nbsp; Trained on 375K+ Images
</div>
""", unsafe_allow_html=True)
