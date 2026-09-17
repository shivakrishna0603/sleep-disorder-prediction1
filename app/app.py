# ============================================================
# app.py — Streamlit Web App (v3: model upgrade + new features)
# Sleep Disorder Prediction Using Wearable Sensor Data
# ============================================================

import os
import io
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch

from smartwatch_import import (
    load_records, extract_features, build_model_input, clamp_to_dataset, FEATURE_KEYS
)
from ble_watch import (
    sync_session_available, SyncSession, scan_sync, load_records_from_db,
    default_db_path, CaptureDB,
)

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Sleep Disorder Predictor",
    page_icon="🛌",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Theme / Custom CSS ───────────────────────────────────────────
st.markdown(
    """
<style>
    #MainMenu, footer {visibility: hidden;}
    header[data-testid="stHeader"] { background: transparent; }
    button[data-testid="stBaseButton-headerNoPadding"],
    div[data-testid="collapsedControl"],
    button[data-testid="stExpandSidebarButton"],
    button[data-testid="stSidebarCollapseButton"] {
        visibility: visible !important;
        display: flex !important;
        opacity: 1 !important;
    }
    /* Hide only the Deploy button, not the whole header toolbar */
    button[data-testid="stAppDeployButton"] { display: none !important; }

    :root {
        --ink: #23264A;
        --coral: #FF6B5B;
        --teal: #00C2A8;
        --yellow: #FFC857;
        --violet: #8B7FFF;
        --cream: #FFF9F2;
    }

    .stApp {
        background: linear-gradient(160deg, #FFF9F2 0%, #F3F0FF 55%, #EAFBF7 100%);
    }

    /* Default text everywhere: dark navy, not white */
    .stApp, .stApp p, .stApp li, .stApp span, .stMarkdown, .stMarkdown p,
    label, .stSelectbox label, .stSlider label {
        color: var(--ink) !important;
    }

    .hero {
        background: linear-gradient(120deg, var(--coral) 0%, #FF9466 45%, var(--yellow) 100%);
        border-radius: 28px;
        border: 3px solid var(--ink);
        box-shadow: 8px 8px 0px var(--ink);
        padding: 2.6rem 2rem;
        text-align: center;
        margin-bottom: 2rem;
    }
    .hero h1 { color: white; font-size: 2.5rem; font-weight: 800; margin: 0; letter-spacing: -0.5px; text-shadow: 2px 2px 0 rgba(0,0,0,0.15); }
    .hero p { color: rgba(255,255,255,0.95); font-size: 1.05rem; margin-top: 0.6rem; font-weight: 500; }

    .card {
        background: #FFFFFF;
        border: 2.5px solid var(--ink);
        border-radius: 20px;
        padding: 1.4rem 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 5px 5px 0px rgba(35,38,74,0.12);
        color: var(--ink);
    }
    .card h4 { margin-top: 0; color: var(--ink); font-size: 1.1rem; font-weight: 800; display: flex; align-items: center; gap: 0.5rem; }
    .card p, .card li, .card b { color: var(--ink) !important; }

    section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] li,
    section[data-testid="stSidebar"] .stMarkdown, section[data-testid="stSidebar"] label {
        color: var(--ink) !important;
    }

    .result-box {
        padding: 1.8rem; border-radius: 20px; text-align: center; margin-bottom: 1rem;
        border: 3px solid var(--ink); box-shadow: 6px 6px 0px rgba(35,38,74,0.15);
    }
    .result-box .label { font-size: 1.7rem; font-weight: 800; margin-bottom: 0.2rem; }
    .result-box .sub { font-size: 0.95rem; opacity: 0.85; font-weight: 600; }

    .result-banner {
        display: flex; align-items: center; gap: 1.4rem;
        padding: 1.6rem 2rem; border-radius: 22px;
        border: 3px solid var(--ink); box-shadow: 7px 7px 0px rgba(35,38,74,0.18);
    }
    .result-banner-emoji { font-size: 3.2rem; line-height: 1; }
    .result-banner-label { font-size: 2rem; font-weight: 800; }
    .result-banner-sub { font-size: 1rem; font-weight: 600; opacity: 0.85; margin-top: 0.2rem; }

    .none      { background: #DFFBF3; color: #00816D; }
    .insomnia  { background: #FFF3D6; color: #B4780A; }
    .apnea     { background: #FFE3E0; color: #D53C2D; }

    .advice-box {
        border-radius: 16px; padding: 1rem 1.2rem; font-size: 0.95rem; font-weight: 500;
        background: #F3F0FF; border: 2px solid var(--violet); color: var(--ink);
    }

    .stButton>button, .stDownloadButton>button {
        background: var(--coral);
        color: white; border: 2.5px solid var(--ink); border-radius: 999px;
        padding: 0.7rem 1.3rem; font-weight: 800; font-size: 1rem;
        box-shadow: 4px 4px 0px var(--ink);
        transition: all 0.12s ease;
    }
    .stButton>button:hover, .stDownloadButton>button:hover {
        transform: translate(-2px, -2px); box-shadow: 6px 6px 0px var(--ink);
        background: #FF7F70;
    }
    .stButton>button:active, .stDownloadButton>button:active {
        transform: translate(1px, 1px); box-shadow: 2px 2px 0px var(--ink);
    }

    section[data-testid="stSidebar"] {
        background: #FFF3E9; border-right: 3px solid var(--ink);
    }

    .footer-note { text-align: center; color: #7a7d99; font-size: 0.82rem; margin-top: 1.5rem; font-weight: 500; }

    .badge {
        display: inline-block; padding: 0.35rem 0.9rem; border-radius: 999px;
        font-size: 0.8rem; font-weight: 800; background: white;
        color: var(--coral); border: 2px solid white;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 0.4rem; }
    .stTabs [data-baseweb="tab"] {
        background: white; border: 2px solid var(--ink); border-radius: 999px;
        padding: 0.5rem 1.1rem; font-weight: 700; color: var(--ink);
    }
    .stTabs [aria-selected="true"] {
        background: var(--teal) !important; color: white !important;
    }

    /* Inputs */
    .stSelectbox div[data-baseweb="select"] > div, input[type="number"], input[type="text"] {
        background: white !important; border: 2px solid var(--ink) !important;
        border-radius: 12px !important; color: var(--ink) !important; font-weight: 600;
    }

    /* Sliders */
    div[data-testid="stSlider"] [role="slider"] { background: var(--coral) !important; border: 2px solid var(--ink) !important; }
    div[data-testid="stSlider"] div[data-baseweb="slider"] > div > div { background: var(--teal) !important; }

    /* Metrics */
    div[data-testid="stMetric"] {
        background: white; border: 2px solid var(--ink); border-radius: 16px;
        padding: 0.8rem; box-shadow: 4px 4px 0px rgba(35,38,74,0.12);
    }
    div[data-testid="stMetricValue"] { color: var(--coral) !important; font-weight: 800; }

    /* Dataframe */
    div[data-testid="stDataFrame"] { border: 2px solid var(--ink); border-radius: 14px; overflow: hidden; }
</style>
""",
    unsafe_allow_html=True,
)

# ── Load Model Artifacts ──────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
REPORT_DIR = os.path.join(os.path.dirname(__file__), "..", "report")


@st.cache_resource
def load_artifacts():
    model = pickle.load(open(os.path.join(MODEL_DIR, "best_model.pkl"), "rb"))
    scaler = pickle.load(open(os.path.join(MODEL_DIR, "scaler.pkl"), "rb"))
    le_target = pickle.load(open(os.path.join(MODEL_DIR, "label_encoder.pkl"), "rb"))
    features = pickle.load(open(os.path.join(MODEL_DIR, "feature_names.pkl"), "rb"))
    cat_encoders = pickle.load(open(os.path.join(MODEL_DIR, "cat_encoders.pkl"), "rb"))
    meta_path = os.path.join(MODEL_DIR, "model_meta.pkl")
    meta = pickle.load(open(meta_path, "rb")) if os.path.exists(meta_path) else None
    return model, scaler, le_target, features, cat_encoders, meta


try:
    model, scaler, le_target, feature_names, cat_encoders, model_meta = load_artifacts()
    model_loaded = True
except Exception as e:
    model, scaler, le_target, feature_names, cat_encoders, model_meta = None, None, None, [], {}, None
    model_loaded = False
    st.error(f"⚠️ Model not found. Please run the training script first.\n\nError: {e}")

GENDER_OPTIONS = list(cat_encoders["Gender"].classes_) if model_loaded else ["Female", "Male"]
OCCUPATION_OPTIONS = list(cat_encoders["Occupation"].classes_) if model_loaded else []
BMI_OPTIONS = list(cat_encoders["BMI Category"].classes_) if model_loaded else []

MODEL_NAME = model_meta["model_name"] if model_meta else type(model).__name__
TEST_ACC = model_meta["test_accuracy"] if model_meta else None
CV_ACC = model_meta["cv_accuracy"] if model_meta else None

RESULT_META = {
    "None": {"emoji": "🟢", "css": "none", "color": "#00C2A8",
              "advice": "Your sleep patterns look healthy! Keep up your current routine — "
                        "consistent bedtime, good activity levels, and manageable stress."},
    "Insomnia": {"emoji": "🟠", "css": "insomnia", "color": "#FFC857",
                 "advice": "Signs consistent with insomnia were detected. Consider a consistent "
                           "sleep schedule, reduce screen time before bed, limit caffeine, and "
                           "consult a doctor if symptoms persist."},
    "Sleep Apnea": {"emoji": "🔴", "css": "apnea", "color": "#FF6B5B",
                    "advice": "Signs consistent with sleep apnea were detected. Please consult a "
                              "healthcare professional — a sleep study can give a proper diagnosis."},
}

if "history" not in st.session_state:
    st.session_state.history = []


def encode_and_predict(input_dict):
    """Shared prediction path used by both single and batch prediction."""
    input_df = pd.DataFrame([input_dict])
    for col in feature_names:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[feature_names]
    input_scaled = scaler.transform(input_df)
    pred = model.predict(input_scaled)[0]
    proba = model.predict_proba(input_scaled)[0]
    label = le_target.inverse_transform([pred])[0]
    return label, proba


def build_pdf_report(input_dict_display, label, proba, classes):
    """Generate an in-memory PDF report for a single prediction."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter, topMargin=0.6 * inch, bottomMargin=0.6 * inch)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("TitleX", parent=styles["Title"], textColor=colors.HexColor("#4a3fb5"))
    section_style = ParagraphStyle("SectionX", parent=styles["Heading2"], spaceBefore=14, textColor=colors.HexColor("#2b2b4a"))
    meta = RESULT_META.get(label, RESULT_META["None"])

    story = [
        Paragraph("Sleep Disorder Prediction Report", title_style),
        Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles["Normal"]),
        Spacer(1, 14),
        Paragraph(f"Result: {label}", section_style),
        Paragraph(f"Confidence: {max(proba) * 100:.1f}%", styles["Normal"]),
        Paragraph(meta["advice"], styles["Normal"]),
        Spacer(1, 10),
        Paragraph("Probability Breakdown", section_style),
    ]

    prob_table_data = [["Class", "Probability"]] + [
        [c, f"{p * 100:.1f}%"] for c, p in zip(classes, proba)
    ]
    prob_table = Table(prob_table_data, colWidths=[3 * inch, 2 * inch])
    prob_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#6a5cff")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(prob_table)
    story.append(Spacer(1, 14))

    story.append(Paragraph("Input Data", section_style))
    input_table_data = [["Field", "Value"]] + [[k, str(v)] for k, v in input_dict_display.items()]
    input_table = Table(input_table_data, colWidths=[3 * inch, 2 * inch])
    input_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4facfe")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(input_table)

    story.append(Spacer(1, 18))
    story.append(Paragraph(
        "Disclaimer: This report is generated by a machine learning model for educational "
        "purposes only. It is not a medical diagnosis. Please consult a healthcare "
        "professional for any health concerns.",
        ParagraphStyle("Disclaimer", parent=styles["Italic"], fontSize=8, textColor=colors.grey),
    ))

    doc.build(story)
    buf.seek(0)
    return buf


# ============================================================
# Shared helpers: Live BLE capture + watch-data prediction UI
# ============================================================

def _render_watch_core(feat, hf, missing, key_suffix, src_text, w_gender, w_age,
                       w_occupation, w_bmi, window_days, sp2_val=None,
                       has_watch_data=True):
    """Metrics card + prediction button + result banner (shared by the
    file-upload path and the Live BLE path)."""
    st.markdown('<div class="card"><h4>📊 Metrics extracted from your watch</h4>',
                unsafe_allow_html=True)
    diag_rows = []
    for key in FEATURE_KEYS:
        val = hf.get(key)
        diag_rows.append({"Feature": key, "Value": val if val is not None else "—",
                          "Source": ""})
    st.dataframe(pd.DataFrame(diag_rows), use_container_width=True, hide_index=True)

    if missing:
        st.warning("⚠️ The following features could not be auto-detected and are "
                   "set to defaults in the form above: " + ", ".join(missing))
    st.caption(src_text)
    st.markdown("</div>", unsafe_allow_html=True)

    do_predict = st.button("🔍 Run Sleep Disorder Prediction from Watch Data",
                           use_container_width=True, key=f"w_predict_{key_suffix}")

    if do_predict:
        if not model_loaded:
            st.error("Model not loaded — run the training script first.")
        elif not has_watch_data:
            st.error("No watch data captured/uploaded yet — run a BLE session or upload a file first.")
        else:
            input_dict_enc = {}
            try:
                input_dict_enc["Gender"] = int(cat_encoders["Gender"].transform([w_gender])[0])
                input_dict_enc["Age"] = w_age
                input_dict_enc["Occupation"] = int(cat_encoders["Occupation"].transform([w_occupation])[0])
                input_dict_enc["BMI Category"] = int(cat_encoders["BMI Category"].transform([w_bmi])[0])
            except Exception as enc_err:
                st.error(f"Encoding error: {enc_err}")
            else:
                for k in FEATURE_KEYS:
                    v = hf.get(k)
                    if v is not None:
                        input_dict_enc[k] = v
                display_dict = {
                    "Gender": w_gender, "Age": w_age, "Occupation": w_occupation,
                    "BMI Category": w_bmi,
                    **{k: hf.get(k) for k in FEATURE_KEYS if hf.get(k) is not None},
                    "_spo2": sp2_val,
                }
                display_dict = {k: v for k, v in display_dict.items() if v is not None}

                label, proba = encode_and_predict(input_dict_enc)
                meta_r = RESULT_META.get(label, RESULT_META["None"])
                confidence = max(proba) * 100

                st.session_state.history.append({
                    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    **{k: v for k, v in display_dict.items() if k != "_spo2"},
                    "Prediction": label, "Confidence": f"{confidence:.1f}%",
                    "Source": src_text,
                })

                if label == "None":
                    st.balloons()

                st.markdown(
                    f"""<div class="result-banner {meta_r['css']}">
                        <div class="result-banner-emoji">{meta_r['emoji']}</div>
                        <div class="result-banner-text">
                            <div class="result-banner-label">{label}</div>
                            <div class="result-banner-sub">Predicted with {confidence:.1f}% confidence — {src_text}</div>
                        </div>
                    </div>""",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<div class="advice-box" style="margin-top:1rem;">'
                    f'{meta_r["emoji"]} {meta_r["advice"]}</div>',
                    unsafe_allow_html=True,
                )

                st.markdown("<div style='margin:1.2rem 0 0.6rem'></div>", unsafe_allow_html=True)
                rc1, rc2 = st.columns(2)
                with rc1:
                    st.markdown('<div class="card"><h4>🎯 Confidence</h4>',
                                unsafe_allow_html=True)
                    gauge = go.Figure(go.Indicator(
                        mode="gauge+number", value=confidence,
                        number={"suffix": "%", "font": {"color": "#23264A", "size": 40}},
                        gauge={"axis": {"range": [0, 100], "tickcolor": "#23264A"},
                               "bar": {"color": meta_r["color"]},
                               "bgcolor": "rgba(35,38,74,0.06)",
                               "borderwidth": 2, "bordercolor": "#23264A"},
                        domain={"x": [0, 1], "y": [0, 1]},
                    ))
                    gauge.update_layout(height=200, margin=dict(l=20, r=20, t=10, b=10),
                                         paper_bgcolor="rgba(0,0,0,0)", font={"color": "#23264A"})
                    st.plotly_chart(gauge, use_container_width=True)
                    pdf_buf = build_pdf_report(display_dict, label, proba,
                                               list(le_target.classes_))
                    st.download_button(
                        "📄 Download PDF Report",
                        data=pdf_buf,
                        file_name=f"sleep_report_watch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                    )
                    st.markdown("</div>", unsafe_allow_html=True)
                with rc2:
                    st.markdown('<div class="card"><h4>📊 Probability by Class</h4>',
                                unsafe_allow_html=True)
                    classes = list(le_target.classes_)
                    colors_list = [RESULT_META.get(c, {}).get("color", "#8B7FFF")
                                   for c in classes]
                    bar_fig = go.Figure(go.Bar(
                        x=proba * 100, y=classes, orientation="h",
                        marker_color=colors_list,
                        marker_line_color="#23264A", marker_line_width=1.5,
                        text=[f"{v:.1f}%" for v in proba * 100],
                        textposition="outside",
                        textfont={"color": "#23264A", "size": 13},
                    ))
                    bar_fig.update_layout(
                        xaxis_title="Probability (%)", xaxis_range=[0, 105],
                        height=200, margin=dict(l=10, r=30, t=10, b=10),
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        font={"color": "#23264A"}, showlegend=False,
                    )
                    st.plotly_chart(bar_fig, use_container_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)

                csv_df = pd.DataFrame([
                    {"Feature": k, "Value": display_dict.get(k)} for k in FEATURE_KEYS
                ])
                st.download_button(
                    "⬇️ Download extracted features CSV",
                    data=csv_df.to_csv(index=False),
                    file_name="watch_features.csv",
                    mime="text/csv",
                    use_container_width=True,
                )


def _render_ble_section():
    """Live BLE capture from a Fireboltt 046 / Da Fit watch (laptop adapter)."""
    if "ble_devices" not in st.session_state:
        st.session_state.ble_devices = []
    if "ble_summary" not in st.session_state:
        st.session_state.ble_summary = None
    if "ble_last" not in st.session_state:
        st.session_state.ble_last = ""
    if "ble_addr" not in st.session_state:
        st.session_state.ble_addr = ""
    if "ble_night" not in st.session_state:
        st.session_state.ble_night = None

    with st.expander("🔵 Live capture from your Fireboltt 046 (Bluetooth)", expanded=False):
        if not sync_session_available:
            st.warning("`bleak` is not installed — run `pip install bleak` to use live capture.")
            return

        c1, c2 = st.columns([1, 2])
        with c1:
            if st.button("🔍 Scan for watch", key="ble_scan"):
                with st.spinner("Scanning for Bluetooth watch…"):
                    try:
                        st.session_state.ble_devices = scan_sync(timeout=10.0)
                        st.session_state.ble_last = (f"Found {len(st.session_state.ble_devices)} "
                                                     f"device(s)")
                    except Exception as e:
                        st.session_state.ble_devices = []
                        st.session_state.ble_last = f"Scan failed: {e}"
        with c2:
            if st.session_state.ble_last:
                st.caption(st.session_state.ble_last)

        addr = ""
        if st.session_state.ble_devices:
            dev_opts = {f"{d.name} ({d.address})  RSSI {d.rssi}"
                        + ("  · 0xFEEA" if d.service_feea else ""): d.address
                        for d in st.session_state.ble_devices}
            addr = st.selectbox("Choose your watch", list(dev_opts), key="ble_choose")
            if addr:
                addr = dev_opts[addr]
        manual = st.text_input("…or paste the watch's BLE MAC address", key="ble_manual")
        addr = (addr or "").strip()
        need_addr = addr if addr else (manual.strip() or None)

        if need_addr:
            b1, b2, b3 = st.columns(3)
            with b1:
                if st.button("🔗 Connect & view info", key="ble_info"):
                    with st.spinner("Connecting…"):
                        try:
                            with SyncSession(need_addr) as s:
                                st.session_state.ble_addr = need_addr
                                st.session_state.ble_info = s.device_info
                                bat = s.battery()
                                steps = s.today_steps()
                                st.session_state.ble_battery = bat
                                st.session_state.ble_steps = steps
                                st.session_state.ble_proto = s.protocol
                                st.session_state.ble_last = "Connected ✔"
                        except Exception as e:
                            st.session_state.ble_last = f"Connect failed: {e}"
            with b2:
                dur = st.slider("HR capture (s)", 10, 180, 60, key="ble_dur",
                                help="Each measurement runs one at a time: HR first (this slider), then SpO2, then blood pressure — even if it takes longer.")
            with b3:
                if st.button("❤️ Start live capture", key="ble_live", disabled=st.session_state.get("ble_busy", False)):
                    st.session_state.ble_busy = True
                    try:
                        samples = {}
                        phases = []
                        with st.spinner(f"Capturing one at a time — HR {dur}s, then SpO2, then BP — keep the watch on your wrist…"):
                            def _on_sample(kind, v):
                                samples[kind] = v
                            def _on_phase(name):
                                phases.append(name)
                            with SyncSession(need_addr) as s:
                                s.capture_all(hr_seconds=dur, hr_every=2.5,
                                              on_sample=_on_sample, on_phase=_on_phase)
                                st.session_state.ble_addr = need_addr
                                st.session_state.ble_summary = s.db.summary()
                                st.session_state.ble_steps = s.today_steps()
                        order = " -> ".join(phases) if phases else "capture finished"
                        st.session_state.ble_last = ("Capture finished" + (f" ({order})" if order else "") +
                                                     " — latest: " +
                                                     ", ".join(f"{k}={v}" for k, v in samples.items()))
                    except Exception as e:
                        st.session_state.ble_last = f"Capture failed: {e}"
                    finally:
                        st.session_state.ble_busy = False

            if st.button("🌙 Sync last 2 days (sleep/steps/HR)", key="ble_sync"):
                with st.spinner("Syncing from watch… keep the app open (~20-30s)"):
                    try:
                        night = None
                        with SyncSession(need_addr) as s:
                            def _prog(msg):
                                pass
                            night = s.sync(with_workouts=True, on_progress=_prog)
                            st.session_state.ble_addr = need_addr
                            st.session_state.ble_info = s.device_info
                            st.session_state.ble_summary = s.db.summary()
                        st.session_state.ble_night = night
                        if night:
                            st.session_state.ble_last = (
                                f"Synced ✔  Last night: {night['duration_hrs']} h "
                                f"(quality {night['quality']})")
                        else:
                            st.session_state.ble_last = "Synced ✔ (no sleep segments stored yet — wear it overnight)"
                    except Exception as e:
                        st.session_state.ble_last = f"Sync failed: {e}"

        info = st.session_state.get("ble_info")
        if info:
            st.markdown(
                '<div style="border:2px dashed #23264A;border-radius:14px;padding:0.6rem 1rem;'
                'font-size:0.9rem;">'
                f"<b>{' · '.join(v for v in info.values() if v)}</b>"
                f"&nbsp;&nbsp;·&nbsp;&nbsp; protocol {st.session_state.get('ble_proto','')}"
                + (f"&nbsp;&nbsp;·&nbsp;&nbsp;🔋 battery {st.session_state['ble_battery']}%"
                   if st.session_state.get("ble_battery") else "")
                + (f"&nbsp;&nbsp;·&nbsp;&nbsp;👣 today {st.session_state['ble_steps']} steps"
                   if st.session_state.get("ble_steps") else "")
                + "</div>",
                unsafe_allow_html=True,
            )
        counts = st.session_state.get("ble_summary")
        if counts:
            filled = {k: v for k, v in counts.items() if v}
            st.caption("Captured so far: " + " · ".join(f"{k}={v}" for k, v in filled.items())
                       if filled else "")


# ── Hero Header ───────────────────────────────────────────────
acc_str = f"{TEST_ACC * 100:.1f}%" if TEST_ACC else "N/A"
st.markdown(
    f"""
<div class="hero">
    <h1>🛌 Sleep Disorder Predictor</h1>
    <p>AI-powered insight from your wearable & lifestyle data — Insomnia, Sleep Apnea, or all clear.</p>
    <div style="margin-top:0.9rem;"><span class="badge">Model: {MODEL_NAME} · {acc_str} test accuracy</span></div>
</div>
""",
    unsafe_allow_html=True,
)

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/sleeping.png", width=70)
    st.markdown("### About This App")
    st.markdown(
        """
This app uses **Machine Learning** to predict sleep disorders from health & lifestyle data.

**Disorders Detected**
- 🟢 None
- 🟠 Insomnia
- 🔴 Sleep Apnea

**How to use**
1. Fill in your health data (or upload a CSV for batch predictions)
2. Click **Predict**
3. Review your result, download a PDF report, and check your session history
        """
    )
    st.divider()
    if TEST_ACC:
        st.metric("🎯 Test Accuracy", f"{TEST_ACC*100:.1f}%")
    if CV_ACC:
        st.metric("📐 Cross-Val Accuracy", f"{CV_ACC*100:.1f}%")
    st.caption(f"Best model selected via grid search: **{MODEL_NAME}**")
    st.divider()
    st.markdown("**Dataset:** Sleep Health & Lifestyle Dataset (Kaggle)")
    st.markdown("**Models Tried:** Logistic Regression, Random Forest, Gradient Boosting, SVM, XGBoost")
    st.divider()
    st.caption("⚠️ Educational tool only — not a medical diagnosis.")

# ── Tabs ──────────────────────────────────────────────────────
tab_watch, tab_predict, tab_batch, tab_history, tab_insights, tab_about = st.tabs(
    ["⌚ Smartwatch Data", "🔮 Predict", "📁 Batch Predict", "🕓 History", "📊 Data Insights", "ℹ️ About"]
)

# =================================================================
# SMARTWATCH DATA TAB
# =================================================================
with tab_watch:
    st.markdown("#### ⌚ Real Data From Your Smartwatch")
    st.caption(
        "Get data from your watch two ways: "
        "**(1)** connect it to THIS laptop over Bluetooth for live HR/SpO₂/BP + "
        "last-night sleep sync (Fireboltt 046 / Da Fit), or "
        "**(2)** upload a Health Connect export zip, GOBOULT Fit / Crrepa `.db` "
        "(pulled via adb), or a CSV with step / sleep / HR columns."
    )

    # --- Live BLE section ----------------------------------------------
    _render_ble_section()

    # --- Source upload --------------------------------------------------
    src_col, cfg_col = st.columns([3, 2])
    with src_col:
        watch_file = st.file_uploader(
            "Upload Health Connect ZIP, .db / .sqlite file(s), or a CSV",
            type=["zip", "db", "sqlite", "sqlite3", "csv"],
            accept_multiple_files=True,
            key="watch_file",
        )
    with cfg_col:
        window_days = st.slider("Lookback window (days)", 1, 30, 7)

    # --- Personal info (manual fields the watch never provides) --------
    st.markdown('<div class="card"><h4>👤 Personal Info (you enter these)</h4>',
                unsafe_allow_html=True)
    pc1, pc2, pc3, pc4 = st.columns(4)
    with pc1:
        w_gender = st.selectbox("Gender", GENDER_OPTIONS, key="w_gender")
    with pc2:
        w_age = st.slider("Age", 18, 80, 30, key="w_age")
    with pc3:
        w_occupation = st.selectbox("Occupation", OCCUPATION_OPTIONS, key="w_occ")
    with pc4:
        w_bmi = st.selectbox("BMI Category", BMI_OPTIONS, key="w_bmi")
    st.markdown("</div>", unsafe_allow_html=True)

    # --- BLE captured data -> prediction --------------------------------
    if st.session_state.get("ble_summary") is not None and st.session_state.get("ble_addr"):
        try:
            ble_records = load_records_from_db(default_db_path())
            ble_feat = extract_features(ble_records, window_days=window_days)
            ble_hf, ble_disp, ble_missing = build_model_input(
                ble_feat, w_gender, w_age, w_occupation, w_bmi
            )
            tbls = ", ".join(k for k in ble_records.get("raw", {}).keys()) or "none"
            _render_watch_core(
                ble_feat, ble_hf, ble_missing, key_suffix="ble", src_text="Live BLE capture",
                w_gender=w_gender, w_age=w_age, w_occupation=w_occupation, w_bmi=w_bmi,
                window_days=window_days,
                sp2_val=ble_feat.get("features", {}).get("_spo2"),
                has_watch_data=sum(ble_feat["record_counts"].values()) > 0,
            )
            st.caption(f"Data source: `{default_db_path()}` · detected tables: {tbls}")
        except Exception as ble_err:
            st.warning(f"Could not build prediction from captured data yet: {ble_err}")

    # --- Parse & extract from uploaded file -----------------------------
    if watch_file:
        sources = [f.getvalue() for f in watch_file] if isinstance(watch_file, list) else [watch_file.getvalue()]
        try:
            records = load_records(sources if len(sources) > 1 else sources[0])
            feat = extract_features(records, window_days=window_days)
            hf, display, missing = build_model_input(
                feat, w_gender, w_age, w_occupation, w_bmi
            )
            src_text = "Smartwatch (real data)"
            _render_watch_core(
                feat, hf, missing, key_suffix="up",
                src_text="uploaded file",
                w_gender=w_gender, w_age=w_age, w_occupation=w_occupation, w_bmi=w_bmi,
                window_days=window_days,
                sp2_val=feat.get("features", {}).get("_spo2"),
                has_watch_data=sum(feat["record_counts"].values()) > 0,
            )
            st.caption(f"**Sources detected:** "
                       f"{', '.join(records.get('_filenames', [])) or 'N/A'}  ·  "
                       f"Tables found: {', '.join(k for k in records.get('raw', {}).keys()) or 'none'}")
        except Exception as parse_err:
            st.error(f"Could not parse uploaded file: {parse_err}")
    elif not st.session_state.get("ble_summary"):
        st.info("Connect the watch below (Bluetooth) for a live session, or upload a Health "
                "Connect export zip, GOBOULT Fit .db files, or a CSV."
                "A single live session captures steps / heart rate / SpO₂ / BP now; "
                "**sleep duration & quality** come from the last night you wore the watch, "
                "fetched with **🌙 Sync last 2 days**.")

# =================================================================
# PREDICT TAB
# =================================================================
with tab_predict:
    with st.form("predict_form"):
        st.markdown("#### 📋 Enter Your Health Data")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown('<div class="card"><h4>👤 Personal Info</h4>', unsafe_allow_html=True)
            gender = st.selectbox("Gender", GENDER_OPTIONS)
            age = st.slider("Age", 18, 80, 30)
            occupation = st.selectbox("Occupation", OCCUPATION_OPTIONS)
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="card"><h4>❤️ Health Metrics</h4>', unsafe_allow_html=True)
            sleep_duration = st.slider("Sleep Duration (hrs)", 4.0, 10.0, 7.0, 0.1)
            quality_of_sleep = st.slider("Quality of Sleep (1-10)", 1, 10, 7)
            heart_rate = st.slider("Heart Rate (bpm)", 50, 100, 72)
            systolic_bp = st.slider("Systolic BP (mmHg)", 90, 180, 120)
            diastolic_bp = st.slider("Diastolic BP (mmHg)", 60, 120, 80)
            st.markdown("</div>", unsafe_allow_html=True)

        with col3:
            st.markdown('<div class="card"><h4>🏃 Lifestyle Metrics</h4>', unsafe_allow_html=True)
            physical_activity = st.slider("Physical Activity Level (min/day)", 0, 120, 45)
            stress_level = st.slider("Stress Level (1-10)", 1, 10, 5)
            bmi_category = st.selectbox("BMI Category", BMI_OPTIONS)
            daily_steps = st.slider("Daily Steps", 1000, 20000, 8000, 500)
            st.markdown("</div>", unsafe_allow_html=True)

        submitted = st.form_submit_button("🔍 Predict Sleep Disorder", use_container_width=True)

    if submitted:
        if not model_loaded:
            st.error("Model not loaded. Please train the model first.")
        else:
            gender_enc = int(cat_encoders["Gender"].transform([gender])[0])
            occ_enc = int(cat_encoders["Occupation"].transform([occupation])[0])
            bmi_enc = int(cat_encoders["BMI Category"].transform([bmi_category])[0])

            input_dict = {
                "Gender": gender_enc, "Age": age, "Occupation": occ_enc,
                "Sleep Duration": sleep_duration, "Quality of Sleep": quality_of_sleep,
                "Physical Activity Level": physical_activity, "Stress Level": stress_level,
                "BMI Category": bmi_enc, "Heart Rate": heart_rate, "Daily Steps": daily_steps,
                "Systolic_BP": systolic_bp, "Diastolic_BP": diastolic_bp,
            }
            label, proba = encode_and_predict(input_dict)
            meta = RESULT_META.get(label, RESULT_META["None"])
            confidence = max(proba) * 100

            display_dict = {
                "Gender": gender, "Age": age, "Occupation": occupation,
                "Sleep Duration (hrs)": sleep_duration, "Quality of Sleep": quality_of_sleep,
                "Physical Activity (min/day)": physical_activity, "Stress Level": stress_level,
                "BMI Category": bmi_category, "Heart Rate (bpm)": heart_rate,
                "Daily Steps": daily_steps, "Systolic BP": systolic_bp, "Diastolic BP": diastolic_bp,
            }

            st.session_state.history.append({
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                **display_dict,
                "Prediction": label,
                "Confidence": f"{confidence:.1f}%",
            })

            if label == "None":
                st.balloons()

            st.markdown("<div style='margin: 2rem 0 1rem;'></div>", unsafe_allow_html=True)

            # ── Big result banner ──────────────────────────────
            st.markdown(
                f"""
                <div class="result-banner {meta['css']}">
                    <div class="result-banner-emoji">{meta['emoji']}</div>
                    <div class="result-banner-text">
                        <div class="result-banner-label">{label}</div>
                        <div class="result-banner-sub">Predicted with {confidence:.1f}% confidence</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(f'<div class="advice-box" style="margin-top:1rem;">{meta["emoji"]} {meta["advice"]}</div>', unsafe_allow_html=True)

            st.markdown("<div style='margin: 1.6rem 0 0.6rem;'></div>", unsafe_allow_html=True)
            res_col1, res_col2 = st.columns([1, 1.1])

            with res_col1:
                st.markdown('<div class="card"><h4>🎯 Confidence Level</h4>', unsafe_allow_html=True)
                gauge = go.Figure(go.Indicator(
                    mode="gauge+number", value=confidence,
                    number={"suffix": "%", "font": {"color": "#23264A", "size": 40}},
                    gauge={"axis": {"range": [0, 100], "tickcolor": "#23264A"},
                           "bar": {"color": meta["color"]},
                           "bgcolor": "rgba(35,38,74,0.06)", "borderwidth": 2, "bordercolor": "#23264A"},
                    domain={"x": [0, 1], "y": [0, 1]},
                ))
                gauge.update_layout(height=200, margin=dict(l=20, r=20, t=10, b=10),
                                     paper_bgcolor="rgba(0,0,0,0)", font={"color": "#23264A"})
                st.plotly_chart(gauge, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

                pdf_buf = build_pdf_report(display_dict, label, proba, list(le_target.classes_))
                dl_col, note_col = st.columns([1, 1])
                with dl_col:
                    st.download_button(
                        "📄 Download PDF Report", data=pdf_buf,
                        file_name=f"sleep_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf", use_container_width=True,
                    )
                with note_col:
                    st.markdown(
                        '<div style="text-align:center; padding-top:0.8rem; font-size:0.82rem; color:#7a7d99;">'
                        '✅ Saved to your History tab</div>', unsafe_allow_html=True,
                    )

            with res_col2:
                st.markdown('<div class="card"><h4>📊 Probability by Class</h4>', unsafe_allow_html=True)
                classes = list(le_target.classes_)
                colors_list = [RESULT_META.get(c, {}).get("color", "#8B7FFF") for c in classes]
                bar = go.Figure(go.Bar(
                    x=proba * 100, y=classes, orientation="h", marker_color=colors_list,
                    marker_line_color="#23264A", marker_line_width=1.5,
                    text=[f"{v:.1f}%" for v in proba * 100], textposition="outside",
                    textfont={"color": "#23264A", "size": 13},
                ))
                bar.update_layout(
                    xaxis_title="Probability (%)", xaxis_range=[0, 105],
                    height=200, margin=dict(l=10, r=30, t=10, b=10),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font={"color": "#23264A"}, showlegend=False,
                )
                st.plotly_chart(bar, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

                with st.expander("📋 Review the data you entered"):
                    input_summary_df = pd.DataFrame(
                        [{"Field": k, "Value": v} for k, v in display_dict.items()]
                    )
                    st.dataframe(input_summary_df, use_container_width=True, hide_index=True)



# =================================================================
# BATCH PREDICT TAB
# =================================================================
with tab_batch:
    st.markdown("#### 📁 Batch Predictions from CSV")
    st.caption(
        "Upload a CSV with columns: Gender, Age, Occupation, Sleep Duration, Quality of Sleep, "
        "Physical Activity Level, Stress Level, BMI Category, Heart Rate, Daily Steps, "
        "Systolic_BP, Diastolic_BP — using the same category names shown in the Predict tab "
        "dropdowns (e.g. Gender as 'Male'/'Female', not numbers)."
    )

    template_df = pd.DataFrame([{
        "Gender": "Male", "Age": 30, "Occupation": "Software Engineer",
        "Sleep Duration": 7.0, "Quality of Sleep": 7, "Physical Activity Level": 45,
        "Stress Level": 5, "BMI Category": "Normal", "Heart Rate": 72,
        "Daily Steps": 8000, "Systolic_BP": 120, "Diastolic_BP": 80,
    }])
    st.download_button(
        "⬇️ Download CSV Template", data=template_df.to_csv(index=False),
        file_name="batch_template.csv", mime="text/csv",
    )

    uploaded = st.file_uploader("Upload your CSV", type=["csv"])

    if uploaded is not None and model_loaded:
        try:
            batch_df = pd.read_csv(uploaded)
            required_cols = ["Gender", "Age", "Occupation", "Sleep Duration", "Quality of Sleep",
                              "Physical Activity Level", "Stress Level", "BMI Category",
                              "Heart Rate", "Daily Steps", "Systolic_BP", "Diastolic_BP"]
            missing = [c for c in required_cols if c not in batch_df.columns]
            if missing:
                st.error(f"Missing required columns: {', '.join(missing)}")
            else:
                results = []
                errors = []
                for idx, row in batch_df.iterrows():
                    try:
                        gender_enc = int(cat_encoders["Gender"].transform([row["Gender"]])[0])
                        occ_enc = int(cat_encoders["Occupation"].transform([row["Occupation"]])[0])
                        bmi_enc = int(cat_encoders["BMI Category"].transform([row["BMI Category"]])[0])
                        input_dict = {
                            "Gender": gender_enc, "Age": row["Age"], "Occupation": occ_enc,
                            "Sleep Duration": row["Sleep Duration"], "Quality of Sleep": row["Quality of Sleep"],
                            "Physical Activity Level": row["Physical Activity Level"], "Stress Level": row["Stress Level"],
                            "BMI Category": bmi_enc, "Heart Rate": row["Heart Rate"], "Daily Steps": row["Daily Steps"],
                            "Systolic_BP": row["Systolic_BP"], "Diastolic_BP": row["Diastolic_BP"],
                        }
                        label, proba = encode_and_predict(input_dict)
                        results.append({**row.to_dict(), "Predicted Disorder": label,
                                         "Confidence": f"{max(proba)*100:.1f}%"})
                    except Exception as row_err:
                        errors.append(f"Row {idx + 1}: {row_err}")
                        results.append({**row.to_dict(), "Predicted Disorder": "ERROR", "Confidence": "-"})

                result_df = pd.DataFrame(results)
                st.success(f"✅ Processed {len(result_df)} rows ({len(errors)} error(s))")
                if errors:
                    with st.expander("⚠️ Rows with errors (e.g. unrecognized category values)"):
                        for e in errors:
                            st.write(e)

                st.dataframe(result_df, use_container_width=True)

                dist = result_df["Predicted Disorder"].value_counts()
                dist_fig = go.Figure(go.Bar(
                    x=dist.index, y=dist.values,
                    marker_color=[RESULT_META.get(c, {}).get("color", "#8f6bff") for c in dist.index],
                ))
                dist_fig.update_layout(
                    title="Predicted Class Distribution", height=300,
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={"color": "#23264A"},
                )
                st.plotly_chart(dist_fig, use_container_width=True)

                st.download_button(
                    "⬇️ Download Results CSV", data=result_df.to_csv(index=False),
                    file_name=f"batch_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv", use_container_width=True,
                )
        except Exception as e:
            st.error(f"Could not process the uploaded file: {e}")

# =================================================================
# HISTORY TAB
# =================================================================
with tab_history:
    st.markdown("#### 🕓 Your Prediction History (this session)")
    st.caption("Resets when you close or refresh the browser tab — download it if you want to keep it.")

    if not st.session_state.history:
        st.info("No predictions yet. Run one from the Predict tab and it will show up here.")
    else:
        hist_df = pd.DataFrame(st.session_state.history)
        st.dataframe(hist_df, use_container_width=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.download_button(
                "⬇️ Download History CSV", data=hist_df.to_csv(index=False),
                file_name="prediction_history.csv", mime="text/csv", use_container_width=True,
            )
        with col_b:
            if st.button("🗑️ Clear History", use_container_width=True):
                st.session_state.history = []
                st.rerun()

# =================================================================
# INSIGHTS TAB
# =================================================================
with tab_insights:
    st.markdown("#### 📊 Exploratory Data Analysis & Model Performance")
    st.caption("Generated from the training dataset during model development.")

    insight_files = [
        ("eda_plots.png", "Exploratory Data Analysis"),
        ("correlation_heatmap.png", "Feature Correlation Heatmap"),
        ("feature_importance.png", "Feature Importance"),
        ("model_comparison.png", "Model Comparison (grid-searched)"),
    ]
    for fname, caption in insight_files:
        fpath = os.path.join(REPORT_DIR, fname)
        if os.path.exists(fpath):
            st.markdown(f'<div class="card"><h4>{caption}</h4>', unsafe_allow_html=True)
            st.image(fpath, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

# =================================================================
# ABOUT TAB
# =================================================================
with tab_about:
    params_str = ", ".join(f"{k}={v}" for k, v in model_meta["params"].items()) if model_meta else "N/A"
    st.markdown(
        f"""
<div class="card">
<h4>🛌 About This Project</h4>
Sleep disorders like insomnia and sleep apnea affect millions of people. Early prediction using
wearable sensor and lifestyle data can help avoid expensive clinical tests.

<br><br><b>Features used:</b> Age, Gender, Occupation, Sleep Duration, Quality of Sleep,
Physical Activity Level, Stress Level, BMI Category, Heart Rate, Daily Steps, Blood Pressure.

<br><br><b>Output classes:</b> None, Insomnia, Sleep Apnea.

<br><br><b>Best model:</b> {MODEL_NAME} (selected via grid search across Logistic Regression,
Random Forest, Gradient Boosting, SVM, and XGBoost)
<br><b>Tuned hyperparameters:</b> {params_str}
<br><b>Test accuracy:</b> {acc_str} &nbsp;|&nbsp; <b>5-fold CV accuracy:</b> {f"{CV_ACC*100:.1f}%" if CV_ACC else "N/A"}

<br><br><b>Tech stack:</b> Python, scikit-learn, XGBoost, Streamlit, Plotly, ReportLab.
</div>
""",
        unsafe_allow_html=True,
    )
    st.warning(
        "⚠️ This tool is for educational purposes only. Always consult a healthcare "
        "professional for medical diagnosis."
    )

# ── Footer ────────────────────────────────────────────────────
st.markdown(
    '<p class="footer-note">B.Tech Final Year Mini Project | Sleep Disorder Prediction Using Wearable Sensor Data</p>',
    unsafe_allow_html=True,
)
