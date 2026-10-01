import io
import json
import os
import re
import time
from datetime import timedelta
from pathlib import Path

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

from src.assistant.assistant import SmartEnergyAssistant
from src.assistant.voice import VoiceAssistant

API_URL = os.getenv(
    "SMARTENERGY_API_URL",
    "http://127.0.0.1:8000"
).rstrip("/")

try:  # optional: PDF export (pip install fpdf2)
    from fpdf import FPDF
    PDF_OK = True
except Exception:
    PDF_OK = False

# ============================================================
# SMART ENERGY AI ASSISTANT
# ============================================================

if "smart_energy_assistant" not in st.session_state:
    st.session_state.smart_energy_assistant = SmartEnergyAssistant(
        api_url=API_URL
    )

if "voice_assistant" not in st.session_state:
    st.session_state.voice_assistant = VoiceAssistant(
        language="en-US"
    )

# ============================================================
# Electricity Demand Forecasting and Anomaly Detection for Smart Homes
# (change APP_NAME below to rename the whole app)
# ============================================================

APP_NAME = "Electricity Demand Forecasting and Anomaly Detection for Smart Homes"
ASSISTANT_NAME = "Smart Home Energy Assistant"
TAGLINE = "Know your home's electricity before the bill arrives"

st.set_page_config(
    page_title=f"{APP_NAME}",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="auto",
)

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent if ROOT.name.lower() == "app" else ROOT
DATA_FILE = PROJECT_ROOT / "results" / "anomaly_detection_results.csv"
MODEL_NAME = "Tuned XGBoost"

# ---- user settings (read from the previous run so they apply before the sidebar is drawn) ----
DARK = bool(st.session_state.get("dark_mode", False))
LANG = st.session_state.get("lang", "English")
try:
    _ua = st.context.headers.get("User-Agent", "") or ""
except Exception:
    _ua = ""
IS_MOBILE = any(k in str(_ua) for k in ("Mobi", "Android", "iPhone", "iPad"))
COMPACT = bool(st.session_state.get("compact", IS_MOBILE))

INK = "#e6f1f5" if DARK else "#12303f"
MUTED = "#9db5c2" if DARK else "#5d7686"
GRID = "rgba(230,241,245,.12)" if DARK else "rgba(18,48,63,.10)"
SURFACE = "#14293a" if DARK else "#ffffff"
AXIS_LINE = "#3a5566" if DARK else "#c3d3dc"
TEAL, MINT, AMBER, RED, VIOLET, BLUE = "#0e9f8e", "#2fbf8f", "#f0a92e", "#e35d6a", "#6b6fe0", "#2f6fce"

# Inline illustration (no external image files needed)
HOUSE_SVG = (
    '<svg viewBox="0 0 260 180" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Smart home illustration">'
    '<defs><linearGradient id="hg" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#12a08e"/>'
    '<stop offset="1" stop-color="#2f6fce"/></linearGradient></defs>'
    '<rect x="14" y="160" width="232" height="7" rx="3.5" fill="#cfe3ea"/>'
    '<path d="M48 88 L130 28 L212 88 V158 H48Z" fill="url(#hg)"/>'
    '<path d="M34 92 L130 18 L226 92" fill="none" stroke="#12303f" stroke-width="8" stroke-linecap="round" stroke-linejoin="round"/>'
    '<rect x="112" y="108" width="36" height="50" rx="5" fill="#ffffff"/>'
    '<rect x="66" y="98" width="30" height="28" rx="5" fill="#ffffff" opacity=".92"/>'
    '<rect x="164" y="98" width="30" height="28" rx="5" fill="#ffffff" opacity=".92"/>'
    '<path d="M136 38 l-12 24 h11 l-7 22 l20 -29 h-12 l9 -17z" fill="#f0a92e"/>'
    '<path d="M108 14 q22 -16 44 0" fill="none" stroke="#6b6fe0" stroke-width="4" stroke-linecap="round"/>'
    '<path d="M117 22 q13 -9 26 0" fill="none" stroke="#6b6fe0" stroke-width="4" stroke-linecap="round"/>'
    '<circle cx="130" cy="29" r="3.5" fill="#6b6fe0"/>'
    '<rect x="222" y="118" width="22" height="42" rx="4" fill="#12303f"/><rect x="226" y="123" width="14" height="8" rx="2" fill="#5dffb8"/>'
    '<rect x="226" y="135" width="14" height="3" rx="1.5" fill="#ffffff" opacity=".7"/>'
    '<rect x="226" y="142" width="14" height="3" rx="1.5" fill="#ffffff" opacity=".7"/>'
    '</svg>'
)

# ---- language support (English / Sinhala / Tamil). Please have a native speaker review the translations. ----
LANGS = ["English", "සිංහල", "தமிழ்"]
TX = {
    "tab_home": ("🏠 Overview", "🏠 දළ විශ්ලේෂණය", "🏠 மேலோட்டம்"),
    "tab_pat": ("📊 Usage patterns", "📊 භාවිත රටා", "📊 பயன்பாட்டு முறைகள்"),
    "tab_anom": ("🚨 Anomalies", "🚨 විෂමතා", "🚨 முரண்பாடுகள்"),
    "tab_save": ("💰 Savings", "💰 ඉතිරිකිරීම්", "💰 சேமிப்பு"),
    "tab_ai": ("🤖 AI assistant", "🤖 AI සහායක", "🤖 AI உதவியாளர்"),
    "tab_exp": ("🔎 Explorer", "🔎 ගවේෂකය", "🔎 ஆய்வுக்கருவி"),
    "tab_fl": ("🔮 Forecast lab", "🔮 පුරෝකථන විද්‍යාගාරය", "🔮 முன்கணிப்பு ஆய்வகம்"),
    "tab_cmp": ("🏘️ Compare homes", "🏘️ නිවාස සංසන්දනය", "🏘️ வீடுகள் ஒப்பீடு"),
    "tab_guide": ("📘 Guide", "📘 මාර්ගෝපදේශය", "📘 வழிகாட்டி"),
    "use_data": ("Use your own data", "ඔබේ දත්ත භාවිතා කරන්න", "உங்கள் தரவைப் பயன்படுத்தவும்"),
    "cur_demand": ("Current demand", "වත්මන් ඉල්ලුම", "தற்போதைய தேவை"),
    "next_fc": ("Next-hour forecast", "ඉදිරි පැය පුරෝකථනය", "அடுத்த மணி முன்கணிப்பு"),
    "last24": ("Last 24 hours", "පසුගිය පැය 24", "கடந்த 24 மணி"),
    "peak24": ("24h peak", "පැය 24 උපරිමය", "24 மணி உச்சம்"),
    "anom7": ("Anomalies (7 days)", "විෂමතා (දින 7)", "முரண்பாடுகள் (7 நாட்கள்)"),
    "ai_insights": ("AI insights", "AI අවබෝධ", "AI நுண்ணறிவு"),
    "eff_score": ("Home efficiency score", "නිවසේ කාර්යක්ෂමතා ලකුණු", "வீட்டு செயல்திறன் மதிப்பெண்"),
    "ways_save": ("Ways to save", "ඉතිරි කරන ක්‍රම", "சேமிக்கும் வழிகள்"),
    "online": ("AI services online", "AI සේවා සක්‍රියයි", "AI சேவைகள் இயங்குகின்றன"),
    "offline": ("AI services offline · charts still work", "AI සේවා අක්‍රියයි · ප්‍රස්තාර තවමත් ක්‍රියා කරයි",
                "AI சேவைகள் இணைப்பில் இல்லை · வரைபடங்கள் செயல்படும்"),
    "digest": ("Daily digest", "දෛනික සාරාංශය", "தினசரி சுருக்கம்"),
}


def T(key):
    i = LANGS.index(LANG) if LANG in LANGS else 0
    return TX.get(key, (key, key, key))[i]


# ============================================================
# STYLE  (light surfaces + dark text everywhere, white text only on gradients)
# ============================================================

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@500;700;800&family=DM+Sans:wght@400;500;700&family=Noto+Sans+Sinhala:wght@400;700&family=Noto+Sans+Tamil:wght@400;700&display=swap');
:root{--ink:#12303f;--muted:#5d7686;--teal:#0e9f8e;--mint:#2fbf8f;--amber:#f0a92e;--red:#e35d6a;--violet:#6b6fe0;--line:#d8e5ec;}
html,body,.stApp{font-family:'DM Sans','Noto Sans Sinhala','Noto Sans Tamil',sans-serif;color-scheme:light;}
.stApp{color:var(--ink);background:radial-gradient(circle at 95% 0%,rgba(14,159,142,.10),transparent 28%),radial-gradient(circle at 0% 100%,rgba(107,111,224,.08),transparent 30%),#f4f8fb;}
.stApp header{background:transparent;}
.block-container{max-width:1500px;padding-top:1rem;padding-bottom:2rem;}
.stApp h1,.stApp h2,.stApp h3,.stApp h4{font-family:'Sora','Noto Sans Sinhala','Noto Sans Tamil',sans-serif;color:var(--ink)!important;}

/* ---- force readable text on Streamlit widgets (works in light and dark themes) ---- */
.stApp [data-testid="stMarkdownContainer"] p,.stApp [data-testid="stMarkdownContainer"] li,
.stApp [data-testid="stWidgetLabel"] p,.stApp [data-testid="stWidgetLabel"] label,.stApp label,
.stApp [data-testid="stCaptionContainer"] p,.stApp [data-testid="stMetricLabel"] *,
.stApp [data-testid="stMetricValue"] *,.stApp [data-testid="stMetricDelta"] *,
.stApp [data-testid="stSlider"] *,.stApp [data-testid="stToggle"] *,
.stApp [data-testid="stAlert"] *,.stApp [data-testid="stExpander"] *,
.stApp .stTabs button p{color:var(--ink);}
.stApp [data-testid="stCaptionContainer"] p{color:var(--muted);}
.stApp input,.stApp textarea{color:var(--ink)!important;-webkit-text-fill-color:var(--ink)!important;background:#fff!important;}
.stApp [data-baseweb="input"],.stApp [data-baseweb="select"]>div,.stApp [data-baseweb="base-input"]{background:#fff!important;border-color:var(--line)!important;}
.stApp [data-baseweb="select"] *{color:var(--ink)!important;}
[data-baseweb="popover"] ul,[data-baseweb="popover"] li{background:#fff!important;color:var(--ink)!important;}
[data-baseweb="popover"] li *{color:var(--ink)!important;}
div[data-testid="stMetric"]{background:#fff;border:1px solid var(--line);border-radius:16px;padding:12px 16px;}
.stApp .stButton button,.stApp [data-testid="stDownloadButton"] button{background:#fff;border:1.5px solid var(--teal);border-radius:12px;font-weight:700;}
.stApp .stButton button p,.stApp [data-testid="stDownloadButton"] button p{color:var(--ink)!important;}
.stApp .stButton button:hover,.stApp [data-testid="stDownloadButton"] button:hover{background:#e3f6f3;}
.stApp [data-testid="stFormSubmitButton"] button{background:linear-gradient(120deg,#0e6f78,#12a08e);border:0;}
.stApp [data-testid="stFormSubmitButton"] button p{color:#fff!important;}
.stApp [data-testid="stForm"]{background:#fff;border:1px solid var(--line);border-radius:16px;}
.stApp [data-testid="stPlotlyChart"]{background:#fff;border:1px solid var(--line);border-radius:16px;padding:6px;box-shadow:0 8px 22px rgba(18,48,63,.05);}

/* ---- sidebar ---- */
[data-testid="stSidebar"]{background:linear-gradient(180deg,#ffffff 0%,#e8f6f3 100%);border-right:1px solid var(--line);}
[data-testid="stSidebar"] *{color:var(--ink);}
[data-testid="stFileUploaderDropzone"]{background:#fff!important;border:2px dashed var(--teal)!important;border-radius:16px!important;}
[data-testid="stFileUploaderDropzone"] *{color:var(--ink)!important;}
[data-testid="stFileUploaderDropzone"] button{background:#fff!important;border:1.5px solid var(--teal)!important;}
.brand{border-radius:18px;padding:16px 18px;margin-bottom:14px;background:linear-gradient(120deg,#0b3a4a,#12a08e);}
.brand .bn{font-family:'Sora',sans-serif;font-weight:800;font-size:1rem;line-height:1.3;}
.brand .bs{font-size:.78rem;opacity:.9;margin-top:6px;line-height:1.4;}
.brand .bi{width:90px;margin:0 0 8px;opacity:.95;}
.stApp .brand,.stApp .brand *{color:#fff!important;}
.mini{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin:8px 0;}
.mini div{background:#fff;border:1px solid var(--line);border-radius:12px;padding:8px;text-align:center;font-size:.72rem;color:var(--muted);}
.mini b{display:block;font-family:'Sora',sans-serif;font-size:1rem;color:var(--ink);}

/* ---- hero ---- */
.hero{position:relative;overflow:hidden;border-radius:26px;padding:30px 34px;background:linear-gradient(120deg,#0b3a4a 0%,#0e6f78 55%,#12a08e 100%);box-shadow:0 22px 50px rgba(11,58,74,.22);}
.hero-grid{display:grid;grid-template-columns:minmax(0,1.5fr) minmax(240px,1fr);gap:26px;align-items:center;position:relative;z-index:2;}
.hero h1{font-size:1.75rem;line-height:1.2;margin:10px 0 8px;letter-spacing:-.5px;}
.hero p{max-width:660px;font-size:.95rem;line-height:1.6;margin:0;opacity:.92;}
.hero-art{position:absolute;right:38%;bottom:-6px;width:210px;opacity:.16;z-index:1;pointer-events:none;}
.pill{display:inline-flex;align-items:center;gap:8px;padding:6px 13px;border-radius:999px;font-size:.78rem;font-weight:700;background:rgba(255,255,255,.16);border:1px solid rgba(255,255,255,.32);}
.dot{width:9px;height:9px;border-radius:50%;background:#5dffb8;animation:pulse 1.6s infinite;}
.dot.off{background:#ff9a9a;animation:none;}
@keyframes pulse{0%{box-shadow:0 0 0 0 rgba(93,255,184,.7);}100%{box-shadow:0 0 0 12px rgba(93,255,184,0);}}
.meter{background:rgba(255,255,255,.13);border:1px solid rgba(255,255,255,.26);border-radius:20px;padding:18px 20px;}
.meter .big{font-family:'Sora',sans-serif;font-size:2.7rem;font-weight:800;line-height:1;}
.meter .unit{font-size:1rem;margin-left:4px;opacity:.9;}
.meter .sub{font-size:.82rem;margin-top:6px;opacity:.92;}
.flow{height:26px;margin-top:12px;border-radius:14px;background:repeating-linear-gradient(90deg,rgba(255,255,255,.06) 0 14px,rgba(255,255,255,.4) 14px 20px);background-size:40px 100%;animation:flow 1.4s linear infinite;}
@keyframes flow{to{background-position:40px 0;}}
.spark{position:absolute;border-radius:50%;background:rgba(255,255,255,.08);animation:float 9s ease-in-out infinite;}
.s1{width:190px;height:190px;right:-50px;top:-70px;} .s2{width:110px;height:110px;right:32%;bottom:-50px;animation-delay:2s;}
@keyframes float{0%,100%{transform:translateY(0);}50%{transform:translateY(14px);}}
.stApp .hero,.stApp .hero *{color:#fff!important;}

/* ---- cards ---- */
.kpi{background:#fff;border:1px solid var(--line);border-radius:18px;padding:16px 18px;min-height:118px;box-shadow:0 8px 24px rgba(18,48,63,.05);border-top:4px solid var(--c,var(--teal));}
.kpi .ic{font-size:1.25rem;} .kpi .lb{color:var(--muted);font-size:.8rem;font-weight:500;margin-top:4px;}
.kpi .vl{font-family:'Sora',sans-serif;font-size:1.5rem;font-weight:800;margin-top:2px;color:var(--ink);}
.kpi .nt{color:var(--muted);font-size:.74rem;margin-top:2px;}
.tip{background:#fff;border:1px solid var(--line);border-left:5px solid var(--c,var(--teal));border-radius:14px;padding:14px 16px;height:100%;box-shadow:0 6px 18px rgba(18,48,63,.04);}
.tip b{font-family:'Sora',sans-serif;font-size:.92rem;color:var(--ink);}
.tip div{color:#3f5b6a;font-size:.84rem;line-height:1.55;margin-top:4px;}
.panel{background:#fff;border:1px solid var(--line);border-radius:18px;padding:18px 20px;box-shadow:0 8px 24px rgba(18,48,63,.05);color:var(--ink);}
.row{display:flex;justify-content:space-between;padding:9px 0;border-bottom:1px solid #eef3f6;font-size:.88rem;}
.row:last-child{border-bottom:0;} .row span{color:var(--muted);}
.ring{--p:70;width:112px;height:112px;border-radius:50%;display:grid;place-items:center;margin:8px auto;background:conic-gradient(var(--c,var(--teal)) calc(var(--p)*1%),#e6eef2 0);}
.ring i{font-style:normal;width:86px;height:86px;border-radius:50%;background:#fff;display:grid;place-items:center;font-family:'Sora',sans-serif;font-weight:800;font-size:1.7rem;color:var(--ink);}
.good{color:#12905f!important;font-weight:700;} .warn{color:#b87708!important;font-weight:700;} .bad{color:#d0404f!important;font-weight:700;}
.chead{margin:14px 0 8px;} .chead .ct{font-family:'Sora',sans-serif;font-weight:700;font-size:1.02rem;color:var(--ink);}
.chead .cd{color:var(--muted);font-size:.83rem;line-height:1.5;margin-top:2px;}
.bub{padding:12px 16px;border-radius:16px;margin:8px 0;font-size:.93rem;line-height:1.6;max-width:88%;}
.bub,.bub *{color:var(--ink)!important;}
.bub.u{background:#dff3ef;border:1px solid #b6e0d8;margin-left:auto;}
.bub.a{background:#fff;border:1px solid var(--line);border-left:5px solid var(--teal);}
.stTabs [data-baseweb="tab-list"]{gap:6px;background:#fff;border:1px solid var(--line);padding:6px;border-radius:16px;flex-wrap:wrap;}
.stTabs [data-baseweb="tab"]{height:42px;border-radius:11px;padding:0 16px;font-weight:700;}
.stTabs [aria-selected="true"]{background:linear-gradient(120deg,#0e6f78,#12a08e);}
.stApp .stTabs [aria-selected="true"] p{color:#fff!important;}
.stTabs [data-baseweb="tab-highlight"],.stTabs [data-baseweb="tab-border"]{display:none;}
.foot{text-align:center;color:#6a8291;font-size:.76rem;padding:22px 0 4px;}

/* ---- guide / badges ---- */
.guide-art{background:#fff;border:1px solid var(--line);border-radius:18px;padding:14px;box-shadow:0 8px 24px rgba(18,48,63,.05);text-align:center;}
.guide-art svg{width:100%;max-width:260px;}
.badge{display:inline-block;padding:2px 10px;border-radius:999px;font-size:.72rem;font-weight:700;background:#e3f6f3;color:#0b6f63!important;}
.sev-High{color:#d0404f!important;font-weight:700;} .sev-Medium{color:#b87708!important;font-weight:700;} .sev-Low{color:#12905f!important;font-weight:700;}
@media(max-width:900px){.hero-grid{grid-template-columns:1fr;}.hero h1{font-size:1.4rem;}.hero-art{display:none;}}
@media(prefers-reduced-motion:reduce){.flow,.dot,.spark{animation:none;}}

/* ---- mobile / tablet ---- */
.stTabs [data-baseweb="tab-list"]{overflow-x:auto;scrollbar-width:thin;}
[data-testid="stDataFrame"],[data-testid="stTable"]{overflow-x:auto;max-width:100%;}
.panel,.tip,.kpi{overflow-wrap:anywhere;}
@media(max-width:1000px){
 [data-testid="stHorizontalBlock"]{flex-wrap:wrap!important;row-gap:.7rem;}
 [data-testid="stHorizontalBlock"]>[data-testid="stColumn"],[data-testid="stHorizontalBlock"]>[data-testid="column"]{flex:1 1 calc(50% - 1rem)!important;min-width:calc(50% - 1rem)!important;}
 .stTabs [data-baseweb="tab-list"]{flex-wrap:nowrap!important;}
 .stTabs [data-baseweb="tab"]{white-space:nowrap;flex:0 0 auto;}
}
@media(max-width:640px){
 .block-container{padding:.5rem .6rem 1.5rem!important;}
 [data-testid="stHorizontalBlock"]>[data-testid="stColumn"],[data-testid="stHorizontalBlock"]>[data-testid="column"]{flex:1 1 100%!important;min-width:100%!important;}
 .stTabs [data-baseweb="tab"]{padding:0 12px;}
 .hero{padding:20px 18px;border-radius:20px;} .hero h1{font-size:1.25rem;} .hero p{font-size:.85rem;}
 .meter{padding:14px 16px;} .meter .big{font-size:2.1rem;}
 .kpi{min-height:auto;padding:12px 14px;} .kpi .vl{font-size:1.25rem;}
 .bub{max-width:100%;}
 .stApp input,.stApp textarea,.stApp [data-baseweb="select"] *{font-size:16px!important;}
 .stApp .stButton button,.stApp [data-testid="stDownloadButton"] button,.stApp [data-testid="stFormSubmitButton"] button{min-height:46px;}
 .chead .ct{font-size:.95rem;}
 .ring{width:96px;height:96px;} .ring i{width:72px;height:72px;font-size:1.4rem;}
 .row{font-size:.82rem;gap:10px;}
}
</style>
""",
    unsafe_allow_html=True,
)

if DARK:
    st.markdown(
        """
<style>
:root{--ink:#e6f1f5;--muted:#9db5c2;--line:#2b4453;}
html,body,.stApp{color-scheme:dark;}
.stApp{background:radial-gradient(circle at 95% 0%,rgba(14,159,142,.14),transparent 28%),radial-gradient(circle at 0% 100%,rgba(107,111,224,.12),transparent 30%),#0d1b24;}
.kpi,.tip,.panel,.bub.a,.guide-art,.mini div,div[data-testid="stMetric"],.stApp [data-testid="stForm"],
.stApp [data-testid="stPlotlyChart"],.stTabs [data-baseweb="tab-list"],.stApp [data-testid="stExpander"],.stApp [data-testid="stAlert"]{background:#14293a!important;border-color:#2b4453!important;}
.tip div{color:#b9cfda;}
.ring i{background:#14293a;}
.ring{background:conic-gradient(var(--c,var(--teal)) calc(var(--p)*1%),#2b4453 0);}
.stApp input,.stApp textarea,.stApp [data-baseweb="input"],.stApp [data-baseweb="select"]>div,.stApp [data-baseweb="base-input"]{background:#14293a!important;}
[data-baseweb="popover"] ul,[data-baseweb="popover"] li{background:#14293a!important;}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0f2230 0%,#0d1b24 100%);}
[data-testid="stFileUploaderDropzone"]{background:#14293a!important;}
[data-testid="stFileUploaderDropzone"] button{background:#14293a!important;}
.stApp .stButton button,.stApp [data-testid="stDownloadButton"] button{background:#14293a;}
.stApp .stButton button:hover,.stApp [data-testid="stDownloadButton"] button:hover{background:#1b3a4f;}
.bub.u{background:#1c4a45;border-color:#2b6a62;}
.row{border-bottom-color:#22394a;}
.foot{color:#8fa8b6;}
</style>
""",
        unsafe_allow_html=True,
    )

# ============================================================
# HELPERS
# ============================================================


def style_fig(fig, height=400, hover="x unified"):
    axis = dict(gridcolor=GRID, zeroline=False, showline=True, linecolor=AXIS_LINE, automargin=True,
                tickfont=dict(color=INK, size=10 if COMPACT else 11), title_font=dict(color=INK, size=11 if COMPACT else 12))
    fig.update_layout(
        template="plotly_white", height=min(height, 320) if COMPACT else height,
        margin=dict(l=8, r=10, t=20, b=60) if COMPACT else dict(l=10, r=15, t=30, b=10),
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, hovermode=hover,
        font=dict(family="DM Sans", color=INK, size=11 if COMPACT else 12),
        legend=dict(orientation="h", y=-0.25 if COMPACT else 1.13, x=0, font=dict(color=INK, size=10 if COMPACT else 11)),
        hoverlabel=dict(bgcolor=SURFACE, font=dict(color=INK)),
        dragmode=False if COMPACT else "zoom",  # lets a finger scroll the page instead of dragging the chart
    )
    fig.update_xaxes(**axis)
    fig.update_yaxes(**axis)
    return fig


def show(fig, key=None):
    st.plotly_chart(fig, use_container_width=True, key=key, theme=None, config={"displayModeBar": False})


def chart_head(title, desc, icon="📊"):
    st.markdown(f'<div class="chead"><div class="ct">{icon} {title}</div><div class="cd">{desc}</div></div>', unsafe_allow_html=True)


def kpi(col, icon, label, value, note="", color=TEAL):
    col.markdown(f'<div class="kpi" style="--c:{color}"><div class="ic">{icon}</div><div class="lb">{label}</div>'
                 f'<div class="vl">{value}</div><div class="nt">{note}</div></div>', unsafe_allow_html=True)


def tip(col, icon, title, text, color=TEAL):
    col.markdown(f'<div class="tip" style="--c:{color}"><b>{icon} {title}</b><div>{text}</div></div>', unsafe_allow_html=True)


def lkr(v):
    return "Unavailable" if v is None else f"LKR {v:,.0f}"


def cmp_block(col, title, cur, prev):
    """Compare two periods (energy + peak) with colour-coded change."""
    col.markdown(f"**{title}**")
    if prev.empty or len(prev) < len(cur):
        col.caption("Not enough earlier data to compare.")
        return
    ce, pe = float(cur["demand_kw"].sum()), float(prev["demand_kw"].sum())
    cp, pp = float(cur["demand_kw"].max()), float(prev["demand_kw"].max())
    col.metric("Energy (kWh)", f"{ce:,.1f}", f"{(ce - pe) / pe * 100:+.1f}%" if pe else None, delta_color="inverse")
    col.metric("Peak (kW)", f"{cp:.2f}", f"{(cp - pp) / pp * 100:+.1f}%" if pp else None, delta_color="inverse")


@st.cache_data(ttl=30, show_spinner=False)
def api(method, endpoint, payload_json=None, timeout=10):
    try:
        t0 = time.perf_counter()
        if method == "GET":
            r = requests.get(f"{API_URL}{endpoint}", timeout=timeout)
        else:
            r = requests.post(f"{API_URL}{endpoint}", json=json.loads(payload_json), timeout=timeout)
        r.raise_for_status()
        return r.json(), (time.perf_counter() - t0) * 1000
    except (requests.RequestException, ValueError):
        return None, None


def history_call(endpoint, data):
    values = data["demand_kw"].tail(169).astype(float).tolist()
    if len(values) < 169:
        return None, None
    payload = {"history_end_timestamp": data["timestamp"].iloc[-1].isoformat(), "hourly_consumption_kw": values}
    return api("POST", endpoint, json.dumps(payload), 15)


def bill_for(kwh):
    res, _ = api("POST", "/cost-estimate", json.dumps({"monthly_energy_kwh": float(round(kwh, 2))}), 10)
    try:
        return float(res["estimated_bill_lkr"]) if res else None
    except (KeyError, TypeError, ValueError):
        return None


def err_stats(actual, pred):
    """MAE, RMSE, MAPE (%) and R2 between actual and predicted values."""
    a, p = np.asarray(actual, dtype=float), np.asarray(pred, dtype=float)
    e = a - p
    mae, rmse = float(np.mean(np.abs(e))), float(np.sqrt(np.mean(e ** 2)))
    mask = a > 0.05
    mape = float(np.mean(np.abs(e[mask]) / a[mask]) * 100) if mask.any() else float("nan")
    ss_tot = float(np.sum((a - a.mean()) ** 2))
    r2 = 1 - float(np.sum(e ** 2)) / ss_tot if ss_tot else float("nan")
    return mae, rmse, mape, r2


def run_backtest(data, n, use_api=True):
    """Replay the last n hours: predict each hour from the 169 hours before it and compare with reality."""
    vals = data["demand_kw"].astype(float).tolist()
    stamps = data["timestamp"].tolist()
    rows = []
    for i in range(len(vals) - n, len(vals)):
        if i < 169:
            continue
        pred = np.nan
        if use_api:
            payload = {"history_end_timestamp": stamps[i - 1].isoformat(), "hourly_consumption_kw": vals[i - 169:i]}
            res, _ = api("POST", "/forecast-from-history", json.dumps(payload), 15)
            try:
                pred = float(res["predicted_demand_kw"])
            except (KeyError, TypeError, ValueError):
                pred = np.nan
        rows.append((stamps[i], vals[i], pred, vals[i - 168], vals[i - 1]))
    return pd.DataFrame(rows, columns=["timestamp", "actual", "predicted", "seasonal", "persistence"])


def _latin(s):
    return str(s).encode("latin-1", "replace").decode("latin-1")


def build_pdf(title, tagline, source, report_df, digest, profile):
    """One-page branded PDF summary (needs fpdf2)."""
    pdf = FPDF(format="A4")
    pdf.set_margins(12, 12, 12)
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_fill_color(11, 58, 74)
    pdf.rect(0, 0, 210, 34, "F")
    pdf.set_xy(12, 8)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 14)
    pdf.multi_cell(186, 6, _latin(title), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)
    pdf.multi_cell(186, 5, _latin(tagline), new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(18, 48, 63)
    pdf.set_xy(12, 40)
    pdf.set_font("Helvetica", "", 9)
    pdf.cell(0, 5, _latin(f"Data source: {source}  |  Generated {pd.Timestamp.now():%d %b %Y %H:%M}"), new_x="LMARGIN", new_y="NEXT")

    pdf.ln(3)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Key figures", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    for metric, value in zip(report_df["Metric"], report_df["Value"]):
        pdf.set_fill_color(232, 246, 243)
        pdf.cell(70, 8, _latin(metric), border=1, fill=True)
        pdf.cell(0, 8, _latin(value), border=1, new_x="LMARGIN", new_y="NEXT")

    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Daily digest", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    for line in digest.split("\n"):
        pdf.multi_cell(0, 6, _latin("- " + line), new_x="LMARGIN", new_y="NEXT")

    pdf.ln(4)
    if pdf.get_y() > 200:
        pdf.add_page()
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Typical day (average kW by hour)", new_x="LMARGIN", new_y="NEXT")
    x0, y0, w, h = 12, pdf.get_y() + 2, 186, 45
    vmax = max(float(profile.max()), 1e-6) * 1.1
    pdf.set_draw_color(200, 215, 222)
    pdf.rect(x0, y0, w, h)
    pts = [(x0 + w * i / 23, y0 + h - h * float(profile.iloc[i]) / vmax) for i in range(24)]
    pdf.set_draw_color(14, 159, 142)
    pdf.set_line_width(0.8)
    for p1, p2 in zip(pts, pts[1:]):
        pdf.line(p1[0], p1[1], p2[0], p2[1])
    pdf.set_line_width(0.2)
    pdf.set_xy(12, y0 + h + 2)
    pdf.set_font("Helvetica", "", 8)
    pdf.cell(0, 5, _latin(f"Hours 0 to 23 left to right. Chart scale goes up to {vmax:.2f} kW."), new_x="LMARGIN", new_y="NEXT")

    pdf.set_y(-18)
    pdf.set_font("Helvetica", "I", 8)
    pdf.cell(0, 5, "Portfolio / academic ML project. Figures are estimates, not a certified meter reading or bill.", align="C")
    return bytes(pdf.output())


# ============================================================
# DATA: built-in dataset + user uploads (CSV / Excel / TXT)
# ============================================================


@st.cache_data
def load_default():
    if not DATA_FILE.exists():
        return pd.DataFrame()
    try:
        d = pd.read_csv(DATA_FILE)
    except Exception:
        return pd.DataFrame()
    if not {"timestamp", "demand_kw"}.issubset(d.columns):
        return pd.DataFrame()
    d["timestamp"] = pd.to_datetime(d["timestamp"], errors="coerce")
    d["demand_kw"] = pd.to_numeric(d["demand_kw"], errors="coerce")
    d["is_anomaly"] = d["is_anomaly"].astype(str).str.lower().isin(["true", "1", "yes", "y"]) if "is_anomaly" in d.columns else False
    if "anomaly_category" not in d.columns:
        d["anomaly_category"] = "Potential anomaly"
    return d.dropna(subset=["timestamp", "demand_kw"]).sort_values("timestamp").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def read_raw(content, name):
    bio = io.BytesIO(content)
    if name.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(bio)
    return pd.read_csv(bio, sep=None, engine="python")


def guess_columns(raw):
    cols = list(raw.columns)
    low = {c: str(c).lower() for c in cols}
    ts = next((c for c in cols if any(k in low[c] for k in ("timestamp", "datetime", "date_time"))), None)
    ts = ts or next((c for c in cols if "date" in low[c] or "time" in low[c]), cols[0])
    numeric = [c for c in cols if c != ts and pd.to_numeric(raw[c], errors="coerce").notna().mean() > .8]
    val = next((c for c in numeric if any(k in low[c] for k in ("demand", "kw", "power", "consum", "load", "usage", "energy"))),
               numeric[0] if numeric else None)
    tcol = next((c for c in cols if c not in (ts, val) and "time" in low[c] and "date" in low[ts]), None)
    return ts, val, tcol


def detect_anomalies(d, threshold):
    s, hr = d["demand_kw"], d["timestamp"].dt.hour
    med = s.groupby(hr).transform("median")
    mad = (s - med).abs().groupby(hr).transform("median").replace(0, np.nan)
    z = (0.6745 * (s - med) / mad).fillna(0)
    d["is_anomaly"] = z.abs() > threshold
    d["anomaly_category"] = np.where(z > 0, "Unusual spike", "Unusual drop")
    return d


@st.cache_data(show_spinner=False)
def process_upload(raw, ts_col, time_col, val_col, unit, threshold):
    d = raw.copy()
    stamp = d[ts_col].astype(str) + ((" " + d[time_col].astype(str)) if time_col else "")
    t = pd.to_datetime(stamp, errors="coerce")
    if t.isna().mean() > .5:
        t = pd.to_datetime(stamp, errors="coerce", dayfirst=True)
    d["timestamp"] = t
    d["demand_kw"] = pd.to_numeric(d[val_col], errors="coerce") * {"kW": 1, "W": .001, "Wh (per hour)": .001, "kWh (per hour)": 1}[unit]
    subs = [c for c in ("sub_metering_1", "sub_metering_2", "sub_metering_3") if c in d.columns]
    for c in subs:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["timestamp", "demand_kw"]).drop_duplicates("timestamp").sort_values("timestamp").set_index("timestamp")
    if d.empty:
        return pd.DataFrame(), 0
    hourly = d[["demand_kw"] + subs].resample(pd.Timedelta(hours=1)).mean()
    gaps = int(hourly["demand_kw"].isna().sum())
    hourly["demand_kw"] = hourly["demand_kw"].interpolate(limit=3)
    hourly[subs] = hourly[subs].fillna(0)
    hourly = hourly.dropna(subset=["demand_kw"]).reset_index()
    return detect_anomalies(hourly, threshold), gaps


TEMPLATE_CSV = pd.DataFrame({
    "timestamp": pd.date_range("2025-01-01", periods=24, freq="h").strftime("%Y-%m-%d %H:%M"),
    "demand_kw": [0.4, 0.4, 0.3, 0.3, 0.4, 0.6, 1.1, 1.5, 1.0, 0.8, 0.7, 0.8, 0.9, 0.8, 0.7, 0.8, 1.0, 1.4, 1.9, 2.1, 1.7, 1.2, 0.8, 0.5],
}).to_csv(index=False).encode("utf-8")

default_df = load_default()
df, source_label, upload_msg = default_df, "built-in dataset", None
gaps_filled = 0

with st.sidebar:
    st.markdown(f'<div class="brand"><div class="bi">{HOUSE_SVG}</div><div class="bn">⚡ {APP_NAME}</div><div class="bs">{TAGLINE}</div></div>', unsafe_allow_html=True)
    with st.expander("⚙️ Settings", expanded=False):
        st.selectbox("🌐 Language / භාෂාව / மொழி", LANGS, key="lang")
        st.toggle("🌙 Dark mode", value=False, key="dark_mode")
        st.toggle("📱 Mobile-friendly charts", value=IS_MOBILE, key="compact",
                  help="Shorter charts, legends underneath and touch-friendly scrolling. Switched on automatically for phone browsers.")
    st.markdown(f"#### 📂 {T('use_data')}")
    st.caption("Upload smart-meter readings and every chart, forecast and alert updates instantly.")
    up = st.file_uploader("Drop a file or browse", type=["csv", "xlsx", "xls", "txt", "tsv"],
                          help="CSV, Excel (.xlsx/.xls) or text files with a date/time column and a power or energy column.")
    st.download_button("⬇ Download CSV template", TEMPLATE_CSV, "energy_template.csv", "text/csv", use_container_width=True)

    if up is not None:
        try:
            raw = read_raw(up.getvalue(), up.name)
        except Exception as e:
            raw = None
            st.error(f"Could not read this file. {e}. For Excel files, run: pip install openpyxl")
        if raw is not None and raw.shape[1] >= 2:
            g_ts, g_val, g_t = guess_columns(raw)
            cols = list(raw.columns)
            st.markdown("**Match your columns**")
            ts_col = st.selectbox("Date / time column", cols, index=cols.index(g_ts))
            num_cols = [c for c in cols if c != ts_col]
            val_col = st.selectbox("Power / energy column", num_cols, index=num_cols.index(g_val) if g_val in num_cols else 0)
            time_opts = ["None"] + [c for c in cols if c not in (ts_col, val_col)]
            time_col = st.selectbox("Separate time column (optional)", time_opts, index=time_opts.index(g_t) if g_t in time_opts else 0)
            unit = st.selectbox("Unit of the values", ["kW", "W", "Wh (per hour)", "kWh (per hour)"])
            sens = st.select_slider("Anomaly sensitivity", ["Low", "Medium", "High"], value="Medium")
            up_df, gaps_filled = process_upload(raw, ts_col, None if time_col == "None" else time_col, val_col, unit,
                                                {"Low": 5.0, "Medium": 4.0, "High": 3.0}[sens])
            if len(up_df) >= 24:
                df, source_label = up_df, up.name
                st.success(f"Using {up.name}")
                st.markdown(f'<div class="mini"><div><b>{len(up_df):,}</b>hourly rows</div>'
                            f'<div><b>{(up_df["timestamp"].max() - up_df["timestamp"].min()).days}</b>days</div>'
                            f'<div><b>{gaps_filled}</b>gaps filled</div></div>', unsafe_allow_html=True)
                if len(up_df) < 169:
                    st.warning("AI forecasting needs at least 169 hourly rows (7 days + 1 hour). Charts still work.")
                st.caption("Anomalies for uploaded data are detected locally by comparing each reading with the usual level for that hour.")
            else:
                st.error("Fewer than 24 valid rows found. Check the column choices and unit. Showing the built-in dataset instead.")
        elif raw is not None:
            st.error("The file needs at least two columns: a date/time and a value.")
    else:
        st.info("Showing the built-in dataset. Upload a file to switch.")

    st.markdown("---")
    if st.button("🔄 Refresh data and AI services", use_container_width=True, key="refresh_all"):
        st.cache_data.clear()
        st.rerun()
    with st.expander("❓ Quick help"):
        st.markdown(
            "**1.** Upload a CSV/Excel file, or use the built-in data.  \n"
            "**2.** Explore the tabs: Overview, Patterns, Anomalies, Savings.  \n"
            "**3.** Use **Forecast lab** to look up to 24 hours ahead.  \n"
            "**4.** Ask the assistant a question in plain English.  \n"
            "**5.** Open the **📘 Guide** tab for definitions and FAQs."
        )
    st.caption(f"AI service address: {API_URL}")

if df.empty:
    st.error("No data available. Upload a CSV or Excel file in the sidebar, or place the dataset at:")
    st.code(str(DATA_FILE))
    st.stop()

sig = (source_label, len(df))
if st.session_state.get("sig") != sig:
    st.session_state.sig = sig
    st.session_state.cursor = min(len(df) - 1, 300)

# ============================================================
# METRICS + AI CALLS
# ============================================================

health, health_ms = api("GET", "/health", None, 3)
api_online = health is not None

latest = df.iloc[-1]
now_ts, now_kw = latest["timestamp"], float(latest["demand_kw"])
last24, last168 = df.tail(24), df.tail(168)
energy_24 = float(last24["demand_kw"].sum())
peak_row = last24.loc[last24["demand_kw"].idxmax()]
peak_24, peak_time = float(peak_row["demand_kw"]), peak_row["timestamp"]
avg_kw = float(df["demand_kw"].mean())
anoms = df[df["is_anomaly"]]
anom_n, anom_rate = len(anoms), len(anoms) / len(df) * 100
recent_anoms = int(last168["is_anomaly"].sum())
monthly_kwh = float(df.groupby(df["timestamp"].dt.to_period("M"))["demand_kw"].sum().mean())
common = anoms["anomaly_category"].mode().iloc[0] if anom_n else "None"

fc, fc_ms, an = None, None, None
if api_online and len(df) >= 169:
    fc, fc_ms = history_call("/forecast-from-history", df)
    an, _ = history_call("/anomaly", df)

forecast_kw = None
try:
    forecast_kw = float(fc["predicted_demand_kw"]) if fc else None
except (KeyError, TypeError, ValueError):
    pass
delta = (forecast_kw - now_kw) / now_kw * 100 if forecast_kw is not None and now_kw else None

if delta is None:
    trend = "Unavailable"
elif delta > 10:
    trend = "Sharp rise expected"
elif delta > 3:
    trend = "Rising"
elif delta < -10:
    trend = "Sharp drop expected"
elif delta < -3:
    trend = "Falling"
else:
    trend = "Steady"

hourly_avg = df.groupby(df["timestamp"].dt.hour)["demand_kw"].mean()
top_hours = hourly_avg.sort_values(ascending=False).head(3).index.tolist()
while len(top_hours) < 3:
    top_hours.append(top_hours[-1])
night_ratio = float(hourly_avg.reindex(range(0, 6)).mean() / hourly_avg.mean()) if hourly_avg.mean() else 0
evening_share = float(hourly_avg.reindex(range(18, 23)).sum() / hourly_avg.sum() * 100)
wk = df["timestamp"].dt.dayofweek >= 5
weekend_gap = (df.loc[wk, "demand_kw"].mean() / df.loc[~wk, "demand_kw"].mean() - 1) * 100 if wk.any() and (~wk).any() else 0.0
peak_ratio = peak_24 / max(float(last24["demand_kw"].mean()), 1e-6)
score = int(np.clip(100 - 18 * max(peak_ratio - 1.5, 0) - 25 * max(night_ratio - 0.6, 0) - 3 * recent_anoms, 20, 99))
score_label, score_col = ("Excellent", TEAL) if score >= 80 else ("Fair", AMBER) if score >= 60 else ("Needs attention", RED)

# ---- daily digest (plain-English summary, also used by the assistant and the PDF) ----
prev24 = df.iloc[-48:-24]
d_change = (energy_24 / float(prev24["demand_kw"].sum()) - 1) * 100 if len(prev24) == 24 and float(prev24["demand_kw"].sum()) else None
if night_ratio > .6:
    digest_tip = "Your overnight use is high. Unplug idle devices before bed to cut standby waste."
elif evening_share > 45:
    digest_tip = f"Almost half of your energy is used in the evening. Move laundry or water heating away from {top_hours[0]:02d}:00."
elif recent_anoms:
    digest_tip = "Some readings were unusual this week. Open the Anomalies tab to see when they happened."
else:
    digest_tip = "Your usage looks healthy. Try a small 5% cut this week to build a saving habit."
digest_lines = [
    f"Last 24 hours: {energy_24:.1f} kWh" + (f" ({d_change:+.1f}% vs the day before)." if d_change is not None else "."),
    f"Peak demand was {peak_24:.2f} kW at {peak_time:%H:%M}.",
    f"Unusual readings in the last 7 days: {recent_anoms}.",
    (f"Next-hour forecast: {forecast_kw:.2f} kW ({trend.lower()})." if forecast_kw is not None
     else "Next-hour forecast is unavailable (AI service offline or too little data)."),
    f"Efficiency score: {score}/100 ({score_label}).",
    f"Tip: {digest_tip}",
]
digest_text = "\n".join(digest_lines)

# ============================================================
# HERO
# ============================================================

live_pill = (f'<span class="pill"><span class="dot"></span>{T("online")}</span>' if api_online
             else f'<span class="pill"><span class="dot off"></span>{T("offline")}</span>')
fc_text = f"{forecast_kw:.2f} kW" if forecast_kw is not None else "Unavailable"
st.markdown(
    f"""
<div class="hero"><div class="spark s1"></div><div class="spark s2"></div><div class="hero-art">{HOUSE_SVG}</div>
<div class="hero-grid">
 <div>{live_pill}
  <h1>{APP_NAME}</h1>
  <p>{TAGLINE}. Predicts your next hour of demand, flags unusual consumption and shows where you can save.
  Data source: {source_label} · {df['timestamp'].min():%d %b %Y} to {df['timestamp'].max():%d %b %Y} · {len(df):,} hourly readings (recorded data, not a live meter).</p>
 </div>
 <div class="meter">
  <div style="font-size:.85rem;opacity:.9">Latest reading · {now_ts:%d %b %H:%M}</div>
  <div><span class="big">{now_kw:.2f}</span><span class="unit">kW</span></div>
  <div class="sub">Next hour: <b>{fc_text}</b> · {trend}</div>
  <div class="flow"></div>
 </div>
</div></div>
""",
    unsafe_allow_html=True,
)
st.write("")
st.caption("New here? Open the **📘 Guide** tab for a 2-minute walkthrough, or ask the 🤖 assistant a question.")
if COMPACT:
    st.info("📱 On a phone? Tap the **›** arrow at the top left to open the sidebar and upload your own data or change settings. "
            "Swipe the tab bar sideways to see every tab.")

tab_home, tab_pat, tab_anom, tab_save, tab_ai, tab_exp, tab_fl, tab_cmp, tab_guide = st.tabs(
    [T("tab_home"), T("tab_pat"), T("tab_anom"), T("tab_save"), T("tab_ai"),
     T("tab_exp"), T("tab_fl"), T("tab_cmp"), T("tab_guide")]
)

# ============================================================
# TAB 1 - OVERVIEW
# ============================================================

with tab_home:
    k = st.columns(5)
    kpi(k[0], "⚡", T("cur_demand"), f"{now_kw:.2f} kW", f"{now_ts:%d %b %H:%M}", TEAL)
    kpi(k[1], "🔮", T("next_fc"), fc_text, f"{delta:+.1f}% vs now" if delta is not None else "Service offline or too little data", VIOLET)
    kpi(k[2], "🔋", T("last24"), f"{energy_24:.1f} kWh", "Total energy used", BLUE)
    kpi(k[3], "📈", T("peak24"), f"{peak_24:.2f} kW", f"at {peak_time:%H:%M}", AMBER)
    kpi(k[4], "🚨", T("anom7"), f"{recent_anoms}", f"{anom_rate:.2f}% of all data", RED if recent_anoms else MINT)

    chart_head("Live energy monitor: your home's power use over time",
               "Teal line = electricity demand in kW. Dotted purple line = 6-hour average (the smooth trend). "
               "Amber dot = the moment currently playing. Red dots = unusual readings. Use the controls to pause or change the window.", "⚡")
    left, right = st.columns([2.3, 1])

    with left:
        c1, c2, c3 = st.columns([1, 1, 2])
        play = c1.toggle("Live playback", value=True, key="play")
        window = c2.selectbox("Window", ["24 hours", "72 hours", "7 days"], index=1, key="win")
        speed = c3.select_slider("Playback speed", ["Slow", "Normal", "Fast"], value="Normal", key="speed")
        hours = {"24 hours": 24, "72 hours": 72, "7 days": 168}[window]
        step = {"Slow": 1, "Normal": 2, "Fast": 6}[speed]

        @st.fragment(run_every=2 if play else None)
        def live_chart():
            if st.session_state.get("play", True):
                st.session_state.cursor = (st.session_state.cursor + step) % len(df)
            end = st.session_state.cursor + 1
            w = df.iloc[max(0, end - hours):end].copy()
            w["avg6"] = w["demand_kw"].rolling(6, min_periods=1).mean()
            cur = w.iloc[-1]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=w["timestamp"], y=w["demand_kw"], name="Demand (kW)", mode="lines",
                                     line=dict(color=TEAL, width=3), fill="tozeroy", fillcolor="rgba(14,159,142,.10)"))
            fig.add_trace(go.Scatter(x=w["timestamp"], y=w["avg6"], name="6-hour average", mode="lines",
                                     line=dict(color=VIOLET, width=2, dash="dot")))
            a = w[w["is_anomaly"]]
            if not a.empty:
                fig.add_trace(go.Scatter(x=a["timestamp"], y=a["demand_kw"], name="Unusual reading", mode="markers",
                                         marker=dict(color=RED, size=9, line=dict(color="#fff", width=1.5))))
            fig.add_trace(go.Scatter(x=[cur["timestamp"]], y=[cur["demand_kw"]], name="Now playing", mode="markers",
                                     marker=dict(color=AMBER, size=15, line=dict(color="#fff", width=3))))
            style_fig(fig, 420)
            fig.update_layout(yaxis_title="Demand (kW)", xaxis_title="Date and time")
            show(fig, "live")
            st.caption(f"Now playing: {cur['timestamp']:%d %b %Y %H:%M} · {cur['demand_kw']:.2f} kW")

        live_chart()

    with right:
        st.markdown(
            f'<div class="panel"><b style="font-family:Sora">{T("eff_score")}</b>'
            f'<div class="ring" style="--p:{score};--c:{score_col}"><i>{score}</i></div>'
            f'<div style="text-align:center;font-weight:700;color:{score_col}">{score_label}</div>'
            f'<div style="text-align:center;color:{MUTED};font-size:.76rem;margin-top:4px">Higher is better. Lowered by sharp peaks, high overnight use and recent anomalies.</div></div>',
            unsafe_allow_html=True)
        st.write("")
        if an is None:
            state, cls = "Unavailable", "warn"
        elif an.get("is_anomaly"):
            state, cls = an.get("anomaly_category", "Potential anomaly"), "bad"
        else:
            state, cls = "Normal", "good"
        st.markdown(
            f'<div class="panel"><div class="row"><span>Trend</span><b>{trend}</b></div>'
            f'<div class="row"><span>Latest reading</span><b class="{cls}">{state}</b></div>'
            f'<div class="row"><span>Model</span><b>{MODEL_NAME}</b></div>'
            f'<div class="row"><span>API speed</span><b>{f"{health_ms:.0f} ms" if health_ms else "N/A"}</b></div></div>',
            unsafe_allow_html=True)

    f1, f2 = st.columns([2, 1])
    with f1:
        chart_head("Next-hour prediction: what the AI expects to happen",
                   "Teal line = your last 48 hours. Purple dashed line = the AI's prediction for the coming hour. If it points up, expect higher use soon.", "🔮")
        rec = df.tail(48)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=rec["timestamp"], y=rec["demand_kw"], name="Last 48 hours", mode="lines", line=dict(color=TEAL, width=3)))
        if forecast_kw is not None:
            fig.add_trace(go.Scatter(x=[now_ts, now_ts + pd.Timedelta(hours=1)], y=[now_kw, forecast_kw], name="AI forecast",
                                     mode="lines+markers", line=dict(color=VIOLET, width=3, dash="dash"), marker=dict(size=10)))
        style_fig(fig, 320)
        fig.update_layout(yaxis_title="Demand (kW)", xaxis_title="Date and time")
        show(fig, "fc")
    with f2:
        chart_head("Load level: is the next hour low, moderate or high?",
                   "Green = light use, amber = moderate, red = heavy. The purple bar shows the forecast.", "🎚️")
        if forecast_kw is not None:
            gmax = max(float(df["demand_kw"].quantile(.99)) * 1.25, forecast_kw * 1.25, 1.0)
            g = go.Figure(go.Indicator(
                mode="gauge+number", value=forecast_kw, number={"suffix": " kW", "font": {"size": 30, "family": "Sora", "color": INK}},
                gauge={"axis": {"range": [0, gmax], "tickfont": {"color": INK}}, "bar": {"color": VIOLET, "thickness": .35},
                       "steps": [{"range": [0, gmax * .4], "color": "#d9f3ea"}, {"range": [gmax * .4, gmax * .75], "color": "#fbebc8"},
                                 {"range": [gmax * .75, gmax], "color": "#f9d5d8"}], "borderwidth": 0}))
            style_fig(g, 320, False)
            show(g, "gauge")
        else:
            st.warning("Forecast unavailable. Start the FastAPI server and make sure there are at least 169 hourly rows.")

    st.markdown(f"### 💡 {T('ai_insights')}")
    ins = st.columns(4)
    tip(ins[0], "⏰", "Your busiest hours", f"Demand peaks around <b>{top_hours[0]:02d}:00</b>, then {top_hours[1]:02d}:00 and {top_hours[2]:02d}:00. Shift flexible tasks outside these hours.", AMBER)
    tip(ins[1], "🌆", "Evening load", f"<b>{evening_share:.0f}%</b> of daily energy is used between 18:00 and 22:00.", VIOLET)
    tip(ins[2], "🌙", "Night base load", f"Overnight demand is <b>{night_ratio*100:.0f}%</b> of your average. " + ("Standby devices may be worth checking." if night_ratio > .6 else "That is a healthy base load."), BLUE if night_ratio <= .6 else RED)
    tip(ins[3], "📅", "Weekends", f"Weekend use is <b>{weekend_gap:+.0f}%</b> compared with weekdays.", TEAL)

    st.markdown(f"### 📰 {T('digest')}")
    st.caption("A plain-English summary of your home. Share it with your family or download it.")
    dg1, dg2 = st.columns([3, 1])
    dg1.markdown('<div class="panel">' + "".join(f'<div class="row"><b style="font-weight:500">{ln}</b></div>' for ln in digest_lines) + "</div>",
                 unsafe_allow_html=True)
    dg2.download_button("⬇ Download digest (TXT)", digest_text.encode("utf-8"), "daily_digest.txt", "text/plain", use_container_width=True, key="dl_digest")

    st.markdown("### 🆚 How you compare")
    st.caption("Change against the period just before. For energy and peak, green (down) is good and red (up) means more use.")
    cb = st.columns(2)
    cmp_block(cb[0], "Last 24 hours vs the 24 hours before", last24, df.iloc[-48:-24])
    cmp_block(cb[1], "Last 7 days vs the 7 days before", last168, df.iloc[-336:-168])

# ============================================================
# TAB 2 - PATTERNS
# ============================================================

with tab_pat:
    p1, p2 = st.columns(2)
    with p1:
        chart_head("Daily energy use: total kWh for each of the last 30 days",
                   "Each bar is one day. Taller bars mean more electricity used. The dotted line is your 30-day average.", "📅")
        daily = df.set_index("timestamp")["demand_kw"].resample("D").sum().tail(30).reset_index()
        fig = go.Figure(go.Bar(x=daily["timestamp"], y=daily["demand_kw"], name="Daily energy", marker_color=TEAL))
        fig.add_hline(y=daily["demand_kw"].mean(), line_dash="dot", line_color=AMBER, annotation_text="30-day average", annotation_font_color=INK)
        style_fig(fig, 360, "x")
        fig.update_layout(yaxis_title="Energy (kWh)", xaxis_title="Date")
        show(fig)
    with p2:
        chart_head("Your typical day: average demand for each hour",
                   "Teal line = average use at that hour. Shaded band = the lowest to highest use ever recorded at that hour. Peaks show when your home is busiest.", "🕒")
        prof = df.groupby(df["timestamp"].dt.hour)["demand_kw"].agg(["min", "mean", "max"])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=prof.index, y=prof["max"], line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=prof.index, y=prof["min"], fill="tonexty", fillcolor="rgba(14,159,142,.15)", line=dict(width=0), name="Lowest to highest range"))
        fig.add_trace(go.Scatter(x=prof.index, y=prof["mean"], name="Average", mode="lines+markers", line=dict(color=TEAL, width=3)))
        style_fig(fig, 360)
        fig.update_layout(xaxis_title="Hour of day (0 to 23)", yaxis_title="Demand (kW)")
        show(fig)

    chart_head("Weekly rhythm: which days and hours use the most power",
               "Each square is a weekday and hour. Darker purple = higher average demand, light green = low. Look for dark patches to find your routine peaks.", "🗓️")
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    heat = df.assign(day=df["timestamp"].dt.day_name(), hour=df["timestamp"].dt.hour).groupby(["day", "hour"])["demand_kw"].mean().unstack().reindex(order)
    fig = go.Figure(go.Heatmap(z=heat.values, x=heat.columns, y=heat.index, colorscale=[[0, "#e8f7f3"], [.5, TEAL], [1, "#2b2f9e"]],
                               colorbar=dict(title=dict(text="kW", font=dict(color=INK)), tickfont=dict(color=INK)),
                               hovertemplate="%{y} %{x}:00<br>%{z:.2f} kW<extra></extra>"))
    style_fig(fig, 340, False)
    fig.update_layout(xaxis_title="Hour of day", yaxis_title="Day of week")
    fig.update_yaxes(autorange="reversed")
    show(fig)

    st.markdown("### 🔌 Where the energy goes")
    sub_map = {"sub_metering_1": "Kitchen", "sub_metering_2": "Laundry", "sub_metering_3": "Water heating"}
    measured = [c for c in sub_map if c in df.columns]
    ld = df[["timestamp", "demand_kw"]].copy()
    if measured:
        for c in measured:
            ld[sub_map[c]] = pd.to_numeric(df[c], errors="coerce").fillna(0).clip(lower=0) / 1000.0
        cats = [sub_map[c] for c in measured]
        ld["Other"] = (ld["demand_kw"] - ld[cats].sum(axis=1)).clip(lower=0)
        cats.append("Other")
        st.success("Using measured sub-meter channels.")
    else:
        h, we = ld["timestamp"].dt.hour, ld["timestamp"].dt.dayofweek >= 5
        sh = pd.DataFrame({
            "Always-on": np.full(len(ld), .30),
            "Water heating": .12 + np.where(((h >= 5) & (h <= 8)) | ((h >= 18) & (h <= 21)), .07, 0),
            "Kitchen": .13 + np.where(((h >= 6) & (h <= 9)) | ((h >= 18) & (h <= 21)), .10, 0),
            "Laundry": .08 + np.where(((h >= 9) & (h <= 16)) & we, .10, .03),
            "Cooling / comfort": .37 - np.where(h <= 5, .10, 0) + np.where((h >= 11) & (h <= 17), .06, 0),
        }, index=ld.index).clip(lower=0)
        sh = sh.div(sh.sum(axis=1), axis=0)
        for c in sh.columns:
            ld[c] = ld["demand_kw"] * sh[c]
        cats = list(sh.columns)
        st.info("Estimated split from time-of-day patterns. It is not measured appliance data. Add smart plugs or sub-meters for real readings.")

    colors = dict(zip(cats, [BLUE, AMBER, TEAL, VIOLET, MINT, MUTED]))
    week = ld.tail(168)
    l1, l2 = st.columns([1.8, 1])
    with l1:
        chart_head("Load breakdown: what is using power, hour by hour (last 7 days)",
                   "Colored layers are stacked, so the top edge equals total demand. A thicker layer means that category used more at that time.", "🧩")
        fig = go.Figure()
        for c in cats:
            fig.add_trace(go.Scatter(x=week["timestamp"], y=week[c], name=c, stackgroup="s", mode="lines", line=dict(width=.8, color=colors[c])))
        style_fig(fig, 380)
        fig.update_layout(yaxis_title="Load (kW)", xaxis_title="Date and time")
        show(fig)
    with l2:
        chart_head("Share of energy by category (last 7 days)",
                   "Each slice shows the percentage of the week's electricity used by that category. The centre shows total kWh.", "🥧")
        tot = {c: float(week[c].sum()) for c in cats}
        fig = go.Figure(go.Pie(labels=list(tot), values=list(tot.values()), hole=.62, textinfo="percent", textposition="outside",
                               textfont=dict(color=INK, size=12), marker=dict(colors=[colors[c] for c in tot])))
        fig.add_annotation(text=f"{sum(tot.values()):.0f}<br>kWh", showarrow=False, font=dict(size=20, family="Sora", color=INK))
        style_fig(fig, 380, False)
        fig.update_layout(legend=dict(orientation="h", y=-.08, font=dict(color=INK)))
        show(fig)

    # ---- NEW: monthly trend + load duration curve ----
    n1, n2 = st.columns(2)
    with n1:
        chart_head("Monthly energy: total kWh per month",
                   "Compare months to spot seasonal changes. Partial months at the start or end of the data will look lower.", "🗓️")
        mon = df.set_index("timestamp")["demand_kw"].resample("MS").sum().reset_index()
        fig = go.Figure(go.Bar(x=mon["timestamp"], y=mon["demand_kw"], name="Monthly energy", marker_color=BLUE))
        style_fig(fig, 340, "x")
        fig.update_layout(yaxis_title="Energy (kWh)", xaxis_title="Month")
        show(fig)
    with n2:
        chart_head("Load duration curve: how often demand stays high",
                   "Readings sorted from highest to lowest. A steep start means short, sharp peaks. A flat curve means steady use.", "📉")
        srt = np.sort(df["demand_kw"].to_numpy())[::-1]
        pct_time = np.linspace(0, 100, len(srt))
        fig = go.Figure(go.Scatter(x=pct_time, y=srt, mode="lines", name="Demand", line=dict(color=VIOLET, width=3),
                                   fill="tozeroy", fillcolor="rgba(107,111,224,.10)"))
        style_fig(fig, 340, "x")
        fig.update_layout(xaxis_title="Share of time demand is at or above this level (%)", yaxis_title="Demand (kW)")
        show(fig)

# ============================================================
# TAB 3 - ANOMALIES
# ============================================================

with tab_anom:
    a1, a2, a3 = st.columns(3)
    kpi(a1, "🚨", "Total anomalies", f"{anom_n:,}", f"{anom_rate:.2f}% of readings", RED)
    kpi(a2, "🗓️", "Last 7 days", f"{recent_anoms}", "Recent flags", AMBER)
    kpi(a3, "🏷️", "Most common type", common, "Category", VIOLET)

    b1, b2 = st.columns([1.7, 1])
    with b1:
        chart_head("Unusual readings: last 14 days with anomalies highlighted",
                   "Teal line = demand. Red diamonds = readings the AI marked as unusual for that time. Hover a diamond to see its type.", "🚨")
        rc = df.tail(336)
        fig = go.Figure(go.Scatter(x=rc["timestamp"], y=rc["demand_kw"], name="Demand (kW)", mode="lines", line=dict(color=TEAL, width=2)))
        pts = rc[rc["is_anomaly"]]
        if not pts.empty:
            fig.add_trace(go.Scatter(x=pts["timestamp"], y=pts["demand_kw"], name="Anomaly", mode="markers",
                                     marker=dict(color=RED, size=11, symbol="diamond", line=dict(color="#fff", width=1.5)),
                                     customdata=pts["anomaly_category"], hovertemplate="%{x}<br>%{y:.2f} kW<br>%{customdata}<extra></extra>"))
        style_fig(fig, 400)
        fig.update_layout(yaxis_title="Demand (kW)", xaxis_title="Date and time")
        show(fig)
    with b2:
        chart_head("When anomalies happen: count by hour of day",
                   "Taller bars mean more anomalies at that hour. The darkest bar is your most unusual hour.", "⏱️")
        if anom_n:
            byh = anoms.groupby(anoms["timestamp"].dt.hour).size().reindex(range(24), fill_value=0)
            fig = go.Figure(go.Bar(x=byh.index, y=byh.values, name="Anomalies", marker_color=[RED if v == byh.max() else "#f3b5bb" for v in byh.values]))
            style_fig(fig, 400, "x")
            fig.update_layout(xaxis_title="Hour of day (0 to 23)", yaxis_title="Number of anomalies")
            show(fig)
        else:
            st.success("No anomalies recorded.")

    if anom_n:
        st.markdown("### 🧾 Most recent events")
        t = anoms.tail(8).sort_values("timestamp", ascending=False)[["timestamp", "demand_kw", "anomaly_category"]].copy()
        t["timestamp"] = t["timestamp"].dt.strftime("%d %b %Y %H:%M")
        st.dataframe(t.rename(columns={"timestamp": "Time", "demand_kw": "Demand (kW)", "anomaly_category": "Category"}), use_container_width=True, hide_index=True)
    st.caption("An anomaly is unusual statistical behaviour. It does not confirm an equipment fault.")

    # ---- NEW: alert centre ----
    st.markdown("### 🔔 Alert centre")
    st.caption("Set your own limit. The app tells you how often your home went above it in the last 7 days.")
    ac1, ac2 = st.columns([1, 2])
    with ac1:
        default_lim = max(round(float(df["demand_kw"].quantile(.95)), 1), 0.1)
        lim = st.number_input("Alert me above (kW)", min_value=0.0, value=default_lim, step=0.1, key="alert_lim")
        cat_opts = sorted(anoms["anomaly_category"].unique().tolist()) if anom_n else []
        cat_sel = st.multiselect("Anomaly types to show", cat_opts, default=cat_opts, key="cat_sel")
    with ac2:
        over = last168[last168["demand_kw"] > lim]
        mm = st.columns(3)
        mm[0].metric("Readings above limit (7 days)", f"{len(over)}")
        mm[1].metric("Share of the week", f"{len(over) / max(len(last168), 1) * 100:.1f}%")
        mm[2].metric("Highest reading", f"{last168['demand_kw'].max():.2f} kW")
        if len(over):
            st.warning(f"Demand went above {lim:.1f} kW {len(over)} time(s) in the last 7 days, most often around "
                       f"{int(over['timestamp'].dt.hour.mode().iloc[0]):02d}:00.")
        else:
            st.success(f"No readings above {lim:.1f} kW in the last 7 days.")

    if anom_n:
        st.markdown("### 🔬 Anomaly explorer")
        st.caption("Severity compares each flagged reading with the usual level for that hour: High = more than 150% away, Medium = more than 75%.")
        ev = anoms[anoms["anomaly_category"].isin(cat_sel)].copy()
        if ev.empty:
            st.info("No anomalies match the selected types.")
        else:
            ev["usual"] = ev["timestamp"].dt.hour.map(hourly_avg)
            ev["Deviation"] = (ev["demand_kw"] / ev["usual"].replace(0, np.nan) - 1) * 100
            ev["Severity"] = np.select([ev["Deviation"].abs() > 150, ev["Deviation"].abs() > 75], ["High", "Medium"], "Low")
            sv = st.columns(3)
            for col, lab, colr in zip(sv, ["High", "Medium", "Low"], [RED, AMBER, MINT]):
                kpi(col, "🎯", f"{lab} severity", f"{int((ev['Severity'] == lab).sum()):,}", "events", colr)
            top = ev.reindex(ev["Deviation"].abs().sort_values(ascending=False).index).head(15)
            view = pd.DataFrame({
                "Time": top["timestamp"].dt.strftime("%d %b %Y %H:%M"),
                "Demand (kW)": top["demand_kw"].round(2),
                "Usual for hour (kW)": top["usual"].round(2),
                "Deviation (%)": top["Deviation"].round(0),
                "Type": top["anomaly_category"],
                "Severity": top["Severity"],
            })
            st.dataframe(view, use_container_width=True, hide_index=True)
            export = ev[["timestamp", "demand_kw", "usual", "Deviation", "anomaly_category", "Severity"]].rename(
                columns={"timestamp": "Time", "demand_kw": "Demand (kW)", "usual": "Usual for hour (kW)",
                         "anomaly_category": "Type"})
            st.download_button("⬇ Download all anomalies (CSV)", export.to_csv(index=False).encode("utf-8"),
                               "anomalies.csv", "text/csv", key="dl_anoms")

# ============================================================
# TAB 4 - SAVINGS
# ============================================================

with tab_save:
    s1, s2 = st.columns([1, 2])
    with s1:
        kwh = st.number_input("Monthly consumption (kWh)", min_value=0.0, value=round(monthly_kwh, 0), step=50.0)
        cut = st.slider("Reduce usage by", 0, 40, 10, 5, format="%d%%")
        ef = st.number_input("Grid emission factor (kg CO₂e per kWh)", min_value=0.0, value=0.0, step=0.01,
                             help="Enter a verified factor for your grid to see a carbon estimate.")
    scen_kwh = kwh * (1 - cut / 100)
    base_bill, scen_bill = (bill_for(kwh), bill_for(scen_kwh)) if api_online else (None, None)
    with s2:
        if base_bill is None or scen_bill is None:
            st.warning("The cost service is offline. Start the FastAPI server to see bill estimates.")
        else:
            m = st.columns(3)
            m[0].metric("Current bill", lkr(base_bill))
            m[1].metric("After reduction", lkr(scen_bill))
            m[2].metric("You save", lkr(base_bill - scen_bill), f"{cut}% less energy")
            st.caption("Estimate from the cost API. Actual bills depend on tariff and customer category.")

    if api_online and base_bill is not None:
        chart_head("Bill vs savings: estimated monthly bill as you cut consumption",
                   "Green line = estimated bill at each reduction level. The amber dot is the scenario you chose with the slider. The lower the line, the more you save.", "💰")
        pcts = list(range(0, 45, 5))
        bills = [bill_for(kwh * (1 - p / 100)) for p in pcts]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=pcts, y=bills, mode="lines+markers", name="Estimated bill", line=dict(color=MINT, width=4),
                                 marker=dict(size=9), fill="tozeroy", fillcolor="rgba(47,191,143,.12)"))
        fig.add_trace(go.Scatter(x=[cut], y=[scen_bill], mode="markers", name="Your scenario", marker=dict(color=AMBER, size=18, line=dict(color="#fff", width=3))))
        style_fig(fig, 340)
        fig.update_layout(xaxis_title="Reduction in consumption (%)", yaxis_title="Estimated monthly bill (LKR)")
        show(fig)

    if ef > 0:
        c = st.columns(3)
        c[0].metric("Current CO₂e", f"{kwh * ef:,.1f} kg")
        c[1].metric("Scenario CO₂e", f"{scen_kwh * ef:,.1f} kg")
        c[2].metric("Avoided", f"{(kwh - scen_kwh) * ef:,.1f} kg")

    st.markdown(f"### 🌱 {T('ways_save')}")
    w = st.columns(3)
    tip(w[0], "🕕", "Move flexible loads", f"Run laundry and water heating outside {top_hours[0]:02d}:00 to flatten your peak.", AMBER)
    tip(w[1], "🔌", "Cut standby power", f"Switch off idle devices overnight. Your night base load is {night_ratio*100:.0f}% of average.", BLUE)
    tip(w[2], "🎯", "Set a target", f"A {cut}% cut is about {kwh - scen_kwh:,.0f} kWh less each month.", TEAL)

    # ---- NEW: budget planner ----
    st.markdown("### 🎯 Monthly budget planner")
    st.caption("Enter the most you want to pay each month. The app shows how close you are and how much you would need to cut.")
    bg1, bg2 = st.columns([1, 2])
    budget = bg1.number_input("Your monthly budget (LKR)", min_value=0.0, value=0.0, step=500.0, key="budget")
    with bg2:
        if budget <= 0:
            st.info("Enter a budget on the left to see your progress.")
        elif base_bill is None:
            st.warning("The cost service is offline, so the budget check is unavailable.")
        else:
            st.progress(min(base_bill / budget, 1.0), text=f"Estimated bill is {base_bill / budget * 100:.0f}% of your budget")
            if base_bill <= budget:
                st.success(f"You are within budget with {lkr(budget - base_bill)} to spare.")
            else:
                need = None
                for p in range(5, 65, 5):
                    b = bill_for(kwh * (1 - p / 100))
                    if b is not None and b <= budget:
                        need = p
                        break
                if need:
                    st.warning(f"You are {lkr(base_bill - budget)} over budget. Cutting usage by about **{need}%** should bring the bill within budget.")
                else:
                    st.error(f"You are {lkr(base_bill - budget)} over budget. Even a 60% cut may not be enough, so consider revising the budget.")

    # ---- NEW: appliance cost calculator ----
    st.markdown("### 🔌 Appliance cost calculator")
    st.caption("Find out what one appliance adds to your monthly bill. Typical wattages are approximate, so edit them to match your device label.")
    APPL = {"Air conditioner (1.5 ton)": 1400, "Ceiling fan": 75, "Refrigerator": 150, "Water heater": 2000,
            "Washing machine": 500, "LED bulb": 9, "Television": 100, "Rice cooker": 700, "Electric iron": 1000}
    ap = st.columns(5)
    appl = ap[0].selectbox("Appliance", list(APPL), key="appl")
    watts = ap[1].number_input("Power (W)", min_value=0, value=int(APPL[appl]), step=10, key=f"watts_{appl}")
    qty = ap[2].number_input("How many", min_value=1, value=1, step=1, key="appl_qty")
    hrs = ap[3].number_input("Hours per day", min_value=0.0, max_value=24.0, value=4.0, step=0.5, key="appl_hrs")
    days = ap[4].number_input("Days per month", min_value=1, max_value=31, value=30, step=1, key="appl_days")
    app_kwh = watts * qty * hrs * days / 1000
    extra = None
    if api_online:
        b0, b1 = bill_for(kwh), bill_for(kwh + app_kwh)
        extra = (b1 - b0) if b0 is not None and b1 is not None else None
    am = st.columns(3)
    am[0].metric("Energy per month", f"{app_kwh:,.1f} kWh")
    am[1].metric("Share of your usage", f"{app_kwh / kwh * 100:.1f}%" if kwh else "N/A")
    am[2].metric("Added to bill", lkr(extra) if extra is not None else "Cost service offline")

    # ---- NEW: time-of-use tariff planner ----
    st.markdown("### ⏱️ Time-of-use tariff planner")
    st.caption("Enter the rates from your own utility. The values below are placeholders, not an official tariff. "
               "The planner spreads your monthly kWh over a typical day using your own usage pattern (energy charge only, no fixed fees).")
    tr = st.columns(3)
    r_off = tr[0].number_input("Off-peak rate (LKR per kWh)", min_value=0.0, value=25.0, step=1.0, key="r_off")
    r_day = tr[1].number_input("Day rate (LKR per kWh)", min_value=0.0, value=35.0, step=1.0, key="r_day")
    r_peak = tr[2].number_input("Peak rate (LKR per kWh)", min_value=0.0, value=50.0, step=1.0, key="r_peak")
    th = st.columns(4)
    pk_s = th[0].number_input("Peak starts (hour)", min_value=0, max_value=23, value=18, step=1, key="pk_s")
    pk_e = th[1].number_input("Peak ends (hour)", min_value=0, max_value=23, value=22, step=1, key="pk_e")
    of_s = th[2].number_input("Off-peak starts (hour)", min_value=0, max_value=23, value=22, step=1, key="of_s")
    of_e = th[3].number_input("Off-peak ends (hour)", min_value=0, max_value=23, value=5, step=1, key="of_e")

    def in_rng(hh, s, e):
        return (s <= hh < e) if s <= e else (hh >= s or hh < e)

    tou_hours = list(range(24))
    tou_prof = hourly_avg.reindex(tou_hours).fillna(0.0)
    tou_day = float(tou_prof.sum())
    tou_load = tou_prof * ((kwh / 30.0) / tou_day if tou_day > 0 else 0.0)
    tou_band = ["Peak" if in_rng(hh, pk_s, pk_e) else "Off-peak" if in_rng(hh, of_s, of_e) else "Day" for hh in tou_hours]
    tou_rate = {"Peak": r_peak, "Day": r_day, "Off-peak": r_off}
    tou_cost = [float(tou_load[hh]) * tou_rate[tou_band[hh]] for hh in tou_hours]
    tou_month = sum(tou_cost) * 30
    peak_share = sum(float(tou_load[hh]) for hh in tou_hours if tou_band[hh] == "Peak") / max(float(tou_load.sum()), 1e-9) * 100
    tm = st.columns(3)
    tm[0].metric("Estimated monthly energy charge", lkr(tou_month))
    tm[1].metric("Average rate you pay", f"LKR {tou_month / kwh:,.1f} per kWh" if kwh else "N/A")
    tm[2].metric("Energy used in peak hours", f"{peak_share:.0f}%")
    band_col = {"Peak": RED, "Day": AMBER, "Off-peak": MINT}
    fig = go.Figure(go.Bar(x=tou_hours, y=tou_cost, marker_color=[band_col[b] for b in tou_band], name="Cost per hour",
                           customdata=tou_band, hovertemplate="%{x}:00 (%{customdata})<br>LKR %{y:.1f} per day<extra></extra>"))
    chart_head("Cost by hour of a typical day",
               "Red = peak hours, amber = day rate, green = off-peak. Tall red bars are the hours where shifting usage pays off most.", "💸")
    style_fig(fig, 320, "x")
    fig.update_layout(xaxis_title="Hour of day (0 to 23)", yaxis_title="Cost per day (LKR)")
    show(fig, "tou_chart")

    # ---- NEW: peak-shift simulator ----
    st.markdown("### 🔀 Peak-shift simulator")
    st.caption("What if you ran part of your peak-hour appliances (laundry, water heater, dishwasher) in the off-peak window instead? Total kWh stays the same.")
    shift = st.slider("Move this share of peak-hour energy to off-peak hours", 0, 50, 20, 5, format="%d%%", key="shift_pct")
    pk_list = [hh for hh in tou_hours if tou_band[hh] == "Peak"]
    of_list = [hh for hh in tou_hours if tou_band[hh] == "Off-peak"]
    if not pk_list or not of_list:
        st.info("Set at least one peak hour and one off-peak hour above to run the simulator.")
    else:
        new_load = tou_load.copy()
        moved = float(tou_load[pk_list].sum()) * shift / 100
        for hh in pk_list:
            new_load[hh] = tou_load[hh] * (1 - shift / 100)
        for hh in of_list:
            new_load[hh] = tou_load[hh] + moved / len(of_list)
        new_month = sum(float(new_load[hh]) * tou_rate[tou_band[hh]] for hh in tou_hours) * 30
        sm = st.columns(4)
        sm[0].metric("Monthly cost now", lkr(tou_month))
        sm[1].metric("After shifting", lkr(new_month))
        sm[2].metric("You save", lkr(tou_month - new_month), f"{(tou_month - new_month) / tou_month * 100:.1f}%" if tou_month else None)
        sm[3].metric("Typical peak demand", f"{float(new_load.max()):.2f} kW", f"{float(new_load.max() - tou_load.max()):+.2f} kW", delta_color="inverse")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=tou_hours, y=tou_load.values, name="Now", mode="lines", line=dict(color=TEAL, width=3)))
        fig.add_trace(go.Scatter(x=tou_hours, y=new_load.values, name="After shifting", mode="lines", line=dict(color=AMBER, width=3, dash="dash")))
        style_fig(fig, 320)
        fig.update_layout(xaxis_title="Hour of day (0 to 23)", yaxis_title="Typical load (kW)")
        show(fig, "shift_chart")
        st.info(f"Each kWh moved from peak to off-peak saves about LKR {max(r_peak - r_off, 0):,.0f}. "
                f"Try running flexible appliances between {of_s:02d}:00 and {of_e:02d}:00.")

    # ---- NEW: weekly saving goal, progress and streaks ----
    st.markdown("### 🏅 Weekly saving goal and streaks")
    st.caption("Set a weekly kWh target. Each completed 7-day period under the target extends your streak.")
    n_weeks = min(len(df) // 168, 8)
    if n_weeks < 2:
        st.info("This needs at least 14 days of data.")
    else:
        wk_tot = [float(df["demand_kw"].iloc[len(df) - 168 * (i + 1): len(df) - 168 * i].sum()) for i in range(n_weeks)][::-1]
        goal_default = float(round(float(np.mean(wk_tot[:-1])) * 0.9, 0))
        goal_kwh = st.number_input("Weekly target (kWh)", min_value=1.0, value=max(goal_default, 1.0), step=5.0,
                                   key=f"goal_{source_label}_{len(df)}", help="Starts at 90% of your usual week.")
        cur_wk = wk_tot[-1]
        streak = 0
        for tot_w in reversed(wk_tot):
            if tot_w <= goal_kwh:
                streak += 1
            else:
                break
        st.progress(min(cur_wk / goal_kwh, 1.0), text=f"This week (last 7 days): {cur_wk:,.0f} of {goal_kwh:,.0f} kWh ({cur_wk / goal_kwh * 100:.0f}%)")
        gm = st.columns(3)
        gm[0].metric("Current streak", f"{streak} week(s)")
        gm[1].metric("Best week", f"{min(wk_tot):,.0f} kWh")
        gm[2].metric("Weeks under target", f"{sum(1 for t_ in wk_tot if t_ <= goal_kwh)} of {len(wk_tot)}")
        earned = [("🌱 First win", streak >= 1), ("🔥 Two in a row", streak >= 2), ("🏆 Four-week champion", streak >= 4)]
        st.markdown(" ".join(f'<span class="badge" style="{"" if ok else "opacity:.4"}">{name}</span>' for name, ok in earned), unsafe_allow_html=True)
        fig = go.Figure(go.Bar(x=[f"Week {i + 1}" for i in range(len(wk_tot))], y=wk_tot, name="Weekly energy",
                               marker_color=[MINT if t_ <= goal_kwh else RED for t_ in wk_tot]))
        fig.add_hline(y=goal_kwh, line_dash="dot", line_color=AMBER, annotation_text="Target", annotation_font_color=INK)
        style_fig(fig, 300, "x")
        fig.update_layout(yaxis_title="Energy (kWh)", xaxis_title="Oldest to newest week")
        show(fig, "goal_chart")

# ============================================================
# TAB 5 - AI ASSISTANT
# ============================================================
# ============================================================
# TAB 5 - AI ASSISTANT
# ============================================================

def assistant_reply(q):
    q = q.lower()

    if "peak" in q or "busiest" in q:
        return (
            f"Your highest demand in the last 24 hours was "
            f"**{peak_24:.2f} kW** at {peak_time:%d %b %H:%M}. "
            f"Across all data, demand is usually highest around "
            f"**{top_hours[0]:02d}:00**."
        )

    if "forecast" in q or "next" in q or "predict" in q:
        if forecast_kw is None:
            return (
                "The forecast is unavailable right now. "
                "Start the FastAPI server and make sure the data "
                "has at least 169 hourly rows."
            )

        return (
            f"The AI expects about **{forecast_kw:.2f} kW** next hour, "
            f"**{delta:+.1f}%** vs now. "
            f"Trend: {trend.lower()}."
        )

    if "anomal" in q or "unusual" in q:
        return (
            f"There are **{anom_n:,}** flagged readings "
            f"({anom_rate:.2f}%), **{recent_anoms}** in the last 7 days. "
            f"Most common type: {common}."
        )

    if "average" in q or "avg" in q:
        return (
            f"Your average demand is **{avg_kw:.2f} kW**, "
            f"roughly **{avg_kw * 24 * 30:,.0f} kWh** per month."
        )

    if "cost" in q or "bill" in q:
        b = bill_for(monthly_kwh) if api_online else None

        return (
            f"At about {monthly_kwh:,.0f} kWh a month, "
            f"the estimated bill is **{lkr(b)}**."
            if b
            else "The cost service is offline right now."
        )

    if "save" in q or "reduce" in q or "tip" in q:
        return (
            f"Three quick wins: "
            f"(1) shift laundry and water heating away from "
            f"**{top_hours[0]:02d}:00**, "
            f"(2) check standby devices at night "
            f"(base load is {night_ratio * 100:.0f}% of average), "
            f"(3) aim for a 10% cut, about "
            f"**{monthly_kwh * .1:,.0f} kWh** a month."
        )

    if "weekend" in q:
        return (
            f"Weekend usage is **{weekend_gap:+.1f}%** "
            f"compared with weekdays."
        )

    if "score" in q or "efficien" in q:
        return (
            f"Your efficiency score is **{score}/100 ({score_label})**. "
            f"It starts at 100 and drops for sharp peaks "
            f"(peak is {peak_ratio:.1f}x the 24h average), "
            f"high overnight use ({night_ratio * 100:.0f}% of average) "
            f"and recent anomalies ({recent_anoms})."
        )

    if "yesterday" in q or "compare" in q or "today" in q:
        prev = df.iloc[-48:-24]

        if len(prev) < 24 or not prev["demand_kw"].sum():
            return (
                "There is not enough earlier data to compare "
                "with the previous 24 hours."
            )

        chg = (
            energy_24 /
            float(prev["demand_kw"].sum()) -
            1
        ) * 100

        return (
            f"The last 24 hours used **{energy_24:.1f} kWh**, "
            f"which is **{chg:+.1f}%** compared with the "
            f"24 hours before."
        )

    if "night" in q or "standby" in q or "base load" in q:
        return (
            f"Overnight demand (00:00 to 06:00) is "
            f"**{night_ratio * 100:.0f}%** of your average. "
            + (
                "That is on the high side, so check for "
                "devices left on standby."
                if night_ratio > .6
                else
                "That looks like a healthy base load."
            )
        )

    if "co2" in q or "carbon" in q or "emission" in q:
        if ef > 0:
            return (
                f"With your emission factor, your monthly usage "
                f"of about {monthly_kwh:,.0f} kWh is roughly "
                f"**{monthly_kwh * ef:,.0f} kg CO₂e**."
            )

        return (
            "Enter a verified grid emission factor in the "
            "💰 Savings tab and I can estimate your carbon footprint."
        )

    if "data" in q or "source" in q:
        return (
            f"You are viewing **{source_label}** with "
            f"**{len(df):,}** hourly readings from "
            f"{df['timestamp'].min():%d %b %Y} to "
            f"{df['timestamp'].max():%d %b %Y}."
        )

    if "digest" in q or "summary" in q or "summar" in q:
        return " · ".join(digest_lines)

    return (
        "I can help with: peak hours, forecast, anomalies, "
        "average use, bills, savings tips, weekends, comparing "
        "with yesterday, night usage, carbon, data source, "
        "a daily summary and your efficiency score."
    )


# ------------------------------------------------------------
# Chat state
# ------------------------------------------------------------

GREETING = (
    "assistant",
    "Hi! Tap a question below, type your own question, "
    "or use the microphone."
)

if "chat" not in st.session_state:
    st.session_state.chat = [GREETING]


# ------------------------------------------------------------
# Voice helper
# ------------------------------------------------------------

def process_voice_question():
    """
    Capture a question from the microphone, process it through
    SmartEnergyAssistant, add the conversation to the chat and
    speak the response.
    """

    voice = st.session_state.voice_assistant
    ai = st.session_state.smart_energy_assistant

    with st.spinner("🎙️ Listening..."):
        question = voice.listen(
            timeout=5,
            phrase_time_limit=10,
        )

    if not question:
        response = (
            "I couldn't understand the voice input. "
            "Please try again."
        )

        st.session_state.chat.append(
            ("assistant", response)
        )

        voice.speak(response)

        return

    # --------------------------------------------------------
    # Prepare current dashboard data for the AI assistant
    # --------------------------------------------------------

    context_data = {
        "history": (
            df["demand_kw"]
            .dropna()
            .tail(169)
            .astype(float)
            .tolist()
        ),
        "demand_kw": (
            float(df["demand_kw"].iloc[-1])
            if len(df)
            else None
        ),
        "monthly_kwh": float(monthly_kwh),
    }

    # --------------------------------------------------------
    # Ask the SmartEnergy AI
    # --------------------------------------------------------

    try:
        response = ai.ask(
            question,
            data=context_data,
        )

    except Exception as voice_err:
        # Fallback to the existing dashboard assistant
        response = assistant_reply(question)

    # --------------------------------------------------------
    # Add to chat
    # --------------------------------------------------------

    st.session_state.chat.append(
        ("user", question)
    )

    st.session_state.chat.append(
        ("assistant", response)
    )

    # --------------------------------------------------------
    # Speak the response
    # --------------------------------------------------------

    voice.speak(response)


# ============================================================
# AI ASSISTANT TAB
# ============================================================

with tab_ai:

    st.markdown(
        f"### 🤖 Ask the {ASSISTANT_NAME}"
    )

    st.caption(
        "Ask about your electricity usage, forecasts, anomalies, "
        "costs and energy-saving opportunities."
    )


    # ========================================================
    # VOICE ASSISTANT
    # ========================================================

    voice_col1, voice_col2, voice_col3 = st.columns(
        [1.4, 2.8, 1.4]
    )

    with voice_col1:

        if st.button(
            "🎙️ Ask by Voice",
            use_container_width=True,
            key="voice_ask_button",
        ):

            try:
                process_voice_question()

            except Exception as voice_error:

                error_message = (
                    "Voice input could not be started. "
                    "Please check that your microphone is connected "
                    "and that PyAudio is installed."
                )

                st.session_state.chat.append(
                    ("assistant", error_message)
                )

                st.error(
                    f"{error_message}\n\n"
                    f"Technical detail: {voice_error}"
                )


    with voice_col2:

        st.markdown(
            """
            <div style="
                padding: 10px 14px;
                border-radius: 10px;
                background: rgba(124, 92, 255, 0.08);
                border: 1px solid rgba(124, 92, 255, 0.20);
                text-align: center;
                margin-top: 2px;
            ">
                🎤 <b>Voice Assistant</b><br>
                <span style="font-size: 0.85rem;">
                Click the microphone button and ask your question.
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )


    with voice_col3:

        if st.button(
            "🔊 Test Voice",
            use_container_width=True,
            key="test_voice_button",
        ):

            test_message = (
                "Hello! I am SmartEnergy AI. "
                "Your voice assistant is working."
            )

            try:
                st.session_state.voice_assistant.speak(
                    test_message
                )

                st.success("Voice output is working.")

            except Exception as voice_error:

                st.error(
                    f"Voice output failed: {voice_error}"
                )


    st.markdown("")


    # ========================================================
    # QUICK QUESTIONS
    # ========================================================

    prompts = [
        "When is my peak?",
        "Next-hour forecast",
        "Any anomalies?",
        "How can I save?",
        "My efficiency score",
        "Compare with yesterday",
        "Night usage",
        "What is my average?",
    ]

    new_q = None

    for row_start in range(0, len(prompts), 4):

        quick = st.columns(4)

        for col, ptxt in zip(
            quick,
            prompts[row_start:row_start + 4]
        ):

            if col.button(
                ptxt,
                use_container_width=True,
                key=f"q_{ptxt}",
            ):
                new_q = ptxt


    # ========================================================
    # TEXT QUESTION
    # ========================================================

    with st.form(
        "ask_form",
        clear_on_submit=True,
    ):

        fc1, fc2 = st.columns([6, 1])

        typed = fc1.text_input(
            "Your question",
            placeholder=(
                "e.g. Why is my night usage high?"
            ),
            label_visibility="collapsed",
        )

        sent = fc2.form_submit_button(
            "Ask",
            use_container_width=True,
        )

    if sent and typed.strip():
        new_q = typed.strip()


    # ========================================================
    # PROCESS TEXT QUESTION
    # ========================================================

    if new_q:

        st.session_state.chat.append(
            ("user", new_q)
        )

        st.session_state.chat.append(
            (
                "assistant",
                assistant_reply(new_q),
            )
        )


    # ========================================================
    # CHAT DISPLAY
    # ========================================================

    for role, msg in st.session_state.chat[-8:]:

        html = re.sub(
            r"\*\*(.+?)\*\*",
            r"<b>\1</b>",
            msg,
        )

        bubble_class = (
            "u"
            if role == "user"
            else "a"
        )

        icon = (
            "🙋 "
            if role == "user"
            else "⚡ "
        )

        st.markdown(
            f'<div class="bub {bubble_class}">'
            f'{icon}{html}'
            f'</div>',
            unsafe_allow_html=True,
        )


    # ========================================================
    # CHAT CONTROLS
    # ========================================================

    cc = st.columns([1, 1, 4])

    if cc[0].button(
        "🗑 Clear chat",
        use_container_width=True,
        key="clear_chat",
    ):

        st.session_state.chat = [GREETING]

        st.rerun()


    transcript = "\n\n".join(
        f"{r.upper()}: "
        f"{re.sub(r'[*]{2}', '', m)}"
        for r, m in st.session_state.chat
    )

    cc[1].download_button(
        "⬇ Save chat",
        transcript.encode("utf-8"),
        "chat_transcript.txt",
        "text/plain",
        use_container_width=True,
    )


    # ========================================================
    # MODEL EXPLANATION
    # ========================================================

    chart_head(
        "How the AI decides: which inputs matter most to the forecast",
        "Longer bars = inputs the model relies on more. "
        "The purple bar is the most important: what your home "
        "used one hour ago.",
        "🧠",
    )


    shap = pd.DataFrame(
        {
            "Importance": [
                .4299,
                .0621,
                .0572,
                .0491,
                .0366,
                .0362,
                .0344,
                .0342,
                .0308,
                .0277,
            ],
            "Meaning": [
                "Usage 1 hour ago",
                "Night-time flag",
                "Average of last 3 hours",
                "Same hour last week",
                "Same hour yesterday",
                "Usage 2 hours ago",
                "Hour of day",
                "Hour cycle (cosine)",
                "Hour cycle (sine)",
                "Same hour 3 days ago",
            ],
        }
    ).sort_values("Importance")


    fig = go.Figure(
        go.Bar(
            x=shap["Importance"],
            y=shap["Meaning"],
            orientation="h",
            name="Importance",
            marker_color=[
                VIOLET
                if v == shap["Importance"].max()
                else "#b9bcf2"
                for v in shap["Importance"]
            ],
        )
    )

    style_fig(
        fig,
        380,
        "y",
    )

    fig.update_layout(
        xaxis_title="Importance (mean |SHAP| value)",
        yaxis_title="Model input",
    )

    show(fig)

    st.caption(
        "Global importance across many predictions. "
        "It explains model behaviour in general, "
        "not one specific forecast."
    )


    # ========================================================
    # DOWNLOADABLE ENERGY REPORT
    # ========================================================

    report = pd.DataFrame(
        {
            "Metric": [
                "Data source",
                "Latest demand",
                "Next-hour forecast",
                "24h energy",
                "24h peak",
                "Average demand",
                "Anomalies",
                "Efficiency score",
            ],
            "Value": [
                source_label,
                f"{now_kw:.2f} kW",
                fc_text,
                f"{energy_24:.2f} kWh",
                f"{peak_24:.2f} kW",
                f"{avg_kw:.2f} kW",
                f"{anom_n:,} ({anom_rate:.2f}%)",
                f"{score}/100",
            ],
        }
    )


    d1, d2, d3 = st.columns(3)


    d1.download_button(
        "⬇ Download energy summary (CSV)",
        report.to_csv(index=False).encode("utf-8"),
        "energy_summary.csv",
        "text/csv",
        use_container_width=True,
    )


    report_html = (
        f"<html><head>"
        f"<meta charset='utf-8'>"
        f"<title>{APP_NAME}</title>"
        "<style>"
        "body{font-family:Arial,sans-serif;"
        "color:#12303f;margin:32px}"
        "table{border-collapse:collapse}"
        "td,th{border:1px solid #d8e5ec;"
        "padding:8px 14px;text-align:left}"
        "th{background:#e3f6f3}"
        "</style>"
        f"</head><body>"
        f"<h1>{APP_NAME}</h1>"
        f"<p>{TAGLINE}</p>"
        f"<p>Report generated from: {source_label}</p>"
        f"{report.to_html(index=False)}"
        f"</body></html>"
    )


    d2.download_button(
        "⬇ Download report (HTML)",
        report_html.encode("utf-8"),
        "energy_report.html",
        "text/html",
        use_container_width=True,
    )


    if PDF_OK:

        try:

            pdf_bytes = build_pdf(
                APP_NAME,
                TAGLINE,
                source_label,
                report,
                digest_text,
                hourly_avg
                .reindex(range(24))
                .fillna(0.0),
            )

            d3.download_button(
                "⬇ Download report (PDF)",
                pdf_bytes,
                "energy_report.pdf",
                "application/pdf",
                use_container_width=True,
            )

        except Exception as pdf_err:

            d3.caption(
                f"PDF could not be built: {pdf_err}"
            )

    else:

        d3.caption(
            "PDF export needs one extra package: "
            "pip install fpdf2"
        )

# ============================================================
# TAB 7 - FORECAST LAB (NEW)
# ============================================================

with tab_fl:
    st.markdown("### 🔮 Forecast lab")
    st.caption("Look beyond the next hour. The AI predicts one hour at a time, so longer forecasts feed each prediction back in. "
               "The further ahead you look, the less certain the result. A simple 'same hour last week' baseline is shown for comparison and works even when the AI service is offline.")
    if len(df) < 169:
        st.warning("Forecasting needs at least 169 hourly rows (7 days + 1 hour).")
    else:
        fl1, fl2, fl3 = st.columns([1.5, 1, 1])
        horizon = fl1.slider("Hours ahead", 1, 24, 6, key="fl_h")
        show_base = fl2.toggle("Show baseline", value=True, key="fl_base")
        run = fl3.button("▶ Run AI forecast", use_container_width=True, key="fl_run")

        full = df["demand_kw"].astype(float).tolist()
        base = []
        for _ in range(horizon):
            base.append((full + base)[-168])
        future = [now_ts + pd.Timedelta(hours=i + 1) for i in range(horizon)]

        if run:
            if not api_online:
                st.error("The AI service is offline. Only the baseline is shown. Start the FastAPI server and try again.")
            else:
                preds_new, vals, end = [], full[-169:], now_ts
                with st.spinner("Asking the AI hour by hour..."):
                    for _ in range(horizon):
                        payload = {"history_end_timestamp": end.isoformat(), "hourly_consumption_kw": vals[-169:]}
                        res, _ = api("POST", "/forecast-from-history", json.dumps(payload), 15)
                        try:
                            p = float(res["predicted_demand_kw"])
                        except (KeyError, TypeError, ValueError):
                            break
                        preds_new.append(p)
                        vals.append(p)
                        end = end + pd.Timedelta(hours=1)
                st.session_state.fl_res = {"sig": sig, "h": horizon, "preds": preds_new}
                if len(preds_new) < horizon:
                    st.warning(f"The AI returned {len(preds_new)} of {horizon} hours. Showing what is available.")

        saved = st.session_state.get("fl_res")
        preds = saved["preds"] if saved and saved["sig"] == sig and saved["h"] == horizon else []

        # confidence range from backtest errors (run the backtest below to switch it on)
        bt_saved = st.session_state.get("bt")
        sigma, band_lo, band_hi = None, None, None
        if bt_saved and bt_saved["sig"] == sig:
            res_ = (bt_saved["df"]["actual"] - bt_saved["df"]["predicted"]).dropna()
            if len(res_) >= 8:
                sigma = float(res_.std())
        if preds and sigma is not None:
            half = 1.28 * sigma * np.sqrt(np.arange(1, len(preds) + 1))
            band_hi = (np.array(preds) + half).tolist()
            band_lo = np.maximum(np.array(preds) - half, 0).tolist()

        hist = df.tail(72)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist["timestamp"], y=hist["demand_kw"], name="Last 72 hours", mode="lines", line=dict(color=TEAL, width=3)))
        if show_base:
            fig.add_trace(go.Scatter(x=[now_ts] + future, y=[now_kw] + base, name="Baseline (same hour last week)", mode="lines",
                                     line=dict(color=AMBER, width=2, dash="dot")))
        if band_hi is not None:
            fig.add_trace(go.Scatter(x=future[:len(preds)], y=band_hi, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=future[:len(preds)], y=band_lo, mode="lines", line=dict(width=0), fill="tonexty",
                                     fillcolor="rgba(107,111,224,.16)", name="Likely range (about 80%)"))
        if preds:
            fig.add_trace(go.Scatter(x=[now_ts] + future[:len(preds)], y=[now_kw] + preds, name="AI forecast", mode="lines+markers",
                                     line=dict(color=VIOLET, width=3, dash="dash"), marker=dict(size=8)))
        style_fig(fig, 400)
        fig.update_layout(yaxis_title="Demand (kW)", xaxis_title="Date and time")
        chart_head(f"Next {horizon} hour(s): history, AI forecast and baseline",
                   "Teal = recent history. Purple dashed = AI forecast. Amber dotted = what last week's pattern suggests.", "🔮")
        show(fig, "fl_chart")
        if band_hi is not None:
            st.caption("Shaded band = approximate 80% range built from your backtest errors and widened for each extra hour ahead. It is a rule of thumb, not a guarantee.")
        elif preds:
            st.caption("Tip: run the backtest in **Model performance** below to add a likely-range band around this forecast.")

        use = preds if preds else base
        label = "AI forecast" if preds else "Baseline"
        pk_i = int(np.argmax(use))
        fk = st.columns(3)
        kpi(fk[0], "🔋", f"Expected energy ({label.lower()})", f"{sum(use):.1f} kWh", f"Next {len(use)} hour(s)", BLUE)
        kpi(fk[1], "📈", "Highest expected demand", f"{use[pk_i]:.2f} kW", f"around {future[pk_i]:%d %b %H:%M}", AMBER)
        kpi(fk[2], "📊", "Average expected demand", f"{float(np.mean(use)):.2f} kW", f"{(float(np.mean(use)) / now_kw - 1) * 100:+.0f}% vs now" if now_kw else "", VIOLET)
        if not preds:
            st.info("Press **▶ Run AI forecast** to get the AI prediction. Until then the numbers above come from the baseline.")

        tbl = pd.DataFrame({"Time": [f"{t:%d %b %H:%M}" for t in future], "Baseline (kW)": np.round(base, 2)})
        if preds:
            tbl["AI forecast (kW)"] = pd.Series(np.round(preds, 2))
            tbl["Change vs now (%)"] = pd.Series([(p / now_kw - 1) * 100 if now_kw else np.nan for p in preds]).round(1)
            if band_hi is not None:
                tbl["Likely low (kW)"] = pd.Series(np.round(band_lo, 2))
                tbl["Likely high (kW)"] = pd.Series(np.round(band_hi, 2))
        with st.expander("📋 See the forecast table", expanded=False):
            st.dataframe(tbl, use_container_width=True, hide_index=True)
        st.download_button("⬇ Download forecast (CSV)", tbl.to_csv(index=False).encode("utf-8"), "forecast.csv", "text/csv", key="dl_fc")

    # ---- NEW: model performance (backtest) ----
    st.markdown("### 🎯 Model performance")
    st.caption("A backtest replays recent hours: the AI predicts each hour using only the data before it, and we compare with what really happened. "
               "It also feeds the likely-range band above.")
    if len(df) < 169 + 24:
        st.info("Backtesting needs at least 193 hourly rows.")
    else:
        bt_opts = [n_ for n_ in (24, 48, 72, 96, 168) if n_ <= len(df) - 169] or [24]
        bp1, bp2 = st.columns([2, 1])
        bt_n = bp1.select_slider("Hours to test", options=bt_opts, value=bt_opts[min(1, len(bt_opts) - 1)], key=f"bt_n_{len(df)}")
        if bp2.button("▶ Run backtest", use_container_width=True, key="bt_run"):
            with st.spinner("Replaying the last hours..."):
                bt_new = run_backtest(df, int(bt_n), api_online)
            st.session_state.bt = {"sig": sig, "n": int(bt_n), "df": bt_new}
            st.rerun()
        bt = st.session_state.get("bt")
        if not (bt and bt["sig"] == sig):
            st.info("Press **▶ Run backtest** to score the model. Each hour needs one API call, so 48 hours takes a few seconds.")
        else:
            bdf = bt["df"]
            ai_df = bdf.dropna(subset=["predicted"])
            ai_ok = len(ai_df) >= 5
            if not ai_ok:
                st.warning("The AI service did not answer during the backtest, so only the simple baselines were scored.")
            rows = []
            if ai_ok:
                rows.append((f"AI model ({MODEL_NAME})", *err_stats(ai_df["actual"], ai_df["predicted"])))
            rows.append(("Same hour last week", *err_stats(bdf["actual"], bdf["seasonal"])))
            rows.append(("Same as previous hour", *err_stats(bdf["actual"], bdf["persistence"])))
            mt = pd.DataFrame(rows, columns=["Method", "MAE (kW)", "RMSE (kW)", "MAPE (%)", "R²"])
            top = mt.iloc[0]
            pk = st.columns(4)
            kpi(pk[0], "📏", "MAE", f"{top['MAE (kW)']:.3f} kW", "Average miss per hour", TEAL)
            kpi(pk[1], "📐", "RMSE", f"{top['RMSE (kW)']:.3f} kW", "Penalises big misses", VIOLET)
            kpi(pk[2], "🎯", "MAPE", f"{top['MAPE (%)']:.1f}%" if pd.notna(top["MAPE (%)"]) else "N/A", "Average % error", AMBER)
            kpi(pk[3], "📈", "R²", f"{top['R²']:.2f}" if pd.notna(top["R²"]) else "N/A", "1.00 = perfect fit", BLUE)
            if ai_ok:
                best_base = float(mt.iloc[1:]["MAE (kW)"].min())
                gain = (best_base - float(top["MAE (kW)"])) / best_base * 100 if best_base else 0.0
                (st.success if gain > 0 else st.warning)(
                    f"The AI's average error is {abs(gain):.0f}% {'lower' if gain > 0 else 'higher'} than the best simple baseline over {len(ai_df)} tested hours.")
            st.dataframe(mt.round(3), use_container_width=True, hide_index=True)

            pv = ai_df if ai_ok else bdf
            pcol = "predicted" if ai_ok else "seasonal"
            pl1, pl2 = st.columns([1.7, 1])
            with pl1:
                chart_head("Predicted vs actual over time", "Teal = what your home really used. Purple dashed = what the model expected for that hour.", "🧪")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=pv["timestamp"], y=pv["actual"], name="Actual", mode="lines", line=dict(color=TEAL, width=3)))
                fig.add_trace(go.Scatter(x=pv["timestamp"], y=pv[pcol], name="Predicted" if ai_ok else "Baseline", mode="lines",
                                         line=dict(color=VIOLET, width=2, dash="dash")))
                style_fig(fig, 340)
                fig.update_layout(yaxis_title="Demand (kW)", xaxis_title="Date and time")
                show(fig, "bt_line")
            with pl2:
                chart_head("Accuracy scatter", "Dots on the diagonal are perfect predictions. The closer, the better.", "🎯")
                lim_max = float(max(pv["actual"].max(), pv[pcol].max())) * 1.05
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=pv["actual"], y=pv[pcol], mode="markers", name="Hours", marker=dict(color=VIOLET, size=7, opacity=.7)))
                fig.add_trace(go.Scatter(x=[0, lim_max], y=[0, lim_max], mode="lines", name="Perfect", line=dict(color=AMBER, dash="dot")))
                style_fig(fig, 340, "closest")
                fig.update_layout(xaxis_title="Actual (kW)", yaxis_title="Predicted (kW)")
                show(fig, "bt_scatter")
            st.download_button("⬇ Download backtest (CSV)", bdf.to_csv(index=False).encode("utf-8"), "backtest.csv", "text/csv", key="dl_bt")
            st.caption("Backtests use recent history only, so results can change with the period tested. Treat them as a health check, not a guarantee.")

# ============================================================
# TAB 8 - COMPARE HOMES (NEW)
# ============================================================

with tab_cmp:
    st.markdown("### 🏘️ Compare homes")
    st.caption("Upload one meter file per home to compare them side by side. Your current data is always included. "
               "Each file needs a date/time column and a power or energy column.")
    cm1, cm2 = st.columns([2, 1])
    cmp_files = cm1.file_uploader("Add home files", type=["csv", "xlsx", "xls", "txt", "tsv"], accept_multiple_files=True, key="cmp_files")
    cmp_unit = cm2.selectbox("Unit used in the uploaded files", ["kW", "W", "Wh (per hour)", "kWh (per hour)"], key="cmp_unit")
    homes = {f"Current: {source_label}": df}
    for cf_ in cmp_files or []:
        try:
            rw = read_raw(cf_.getvalue(), cf_.name)
            g_ts2, g_val2, g_t2 = guess_columns(rw)
            if g_val2 is None:
                st.warning(f"{cf_.name}: could not find a numeric power or energy column.")
                continue
            hd, _gaps = process_upload(rw, g_ts2, g_t2, g_val2, cmp_unit, 4.0)
            if len(hd) >= 24:
                homes[cf_.name] = hd
            else:
                st.warning(f"{cf_.name}: fewer than 24 valid hourly rows, so it was skipped.")
        except Exception as cmp_err:
            st.warning(f"{cf_.name}: could not be read ({cmp_err}).")

    if len(homes) < 2:
        st.info("Add at least one more file above to start comparing. Tip: keep the same unit across files, or upload them in separate batches.")
    else:
        rows_c, profs = [], {}
        for hname, hd in homes.items():
            hp = hd.groupby(hd["timestamp"].dt.hour)["demand_kw"].mean().reindex(range(24)).fillna(0.0)
            profs[hname] = hp
            avg_c, peak_c = float(hd["demand_kw"].mean()), float(hd["demand_kw"].max())
            nr_c = float(hp.iloc[0:6].mean() / hp.mean() * 100) if hp.mean() else 0.0
            rows_c.append({
                "Home": hname,
                "Days of data": int((hd["timestamp"].max() - hd["timestamp"].min()).days) + 1,
                "Average (kW)": round(avg_c, 2),
                "Peak (kW)": round(peak_c, 2),
                "Monthly energy (kWh)": round(float(hd.groupby(hd["timestamp"].dt.to_period("M"))["demand_kw"].sum().mean()), 0),
                "Busiest hour": f"{int(hp.idxmax()):02d}:00",
                "Night base load (% of avg)": round(nr_c, 0),
                "Load factor (%)": round(avg_c / peak_c * 100, 0) if peak_c else 0,
                "Anomalies (%)": round(float(hd["is_anomaly"].mean() * 100), 2),
            })
        cmp_df = pd.DataFrame(rows_c)
        best_m = cmp_df.loc[cmp_df["Monthly energy (kWh)"].idxmin()]
        best_n = cmp_df.loc[cmp_df["Night base load (% of avg)"].idxmin()]
        best_l = cmp_df.loc[cmp_df["Load factor (%)"].idxmax()]
        ck = st.columns(3)
        kpi(ck[0], "🏆", "Lowest monthly energy", f"{best_m['Monthly energy (kWh)']:,.0f} kWh", str(best_m["Home"])[:34], MINT)
        kpi(ck[1], "🌙", "Lowest night base load", f"{best_n['Night base load (% of avg)']:.0f}%", str(best_n["Home"])[:34], BLUE)
        kpi(ck[2], "🧮", "Steadiest use (load factor)", f"{best_l['Load factor (%)']:.0f}%", str(best_l["Home"])[:34], VIOLET)
        st.dataframe(cmp_df, use_container_width=True, hide_index=True)

        palette = [TEAL, VIOLET, AMBER, BLUE, RED, MINT, MUTED]
        cc1, cc2 = st.columns([1.6, 1])
        with cc1:
            chart_head("Typical day for each home", "Average demand at each hour. Different shapes show different routines.", "🕒")
            fig = go.Figure()
            for i_, (hname, hp) in enumerate(profs.items()):
                fig.add_trace(go.Scatter(x=list(range(24)), y=hp.values, name=hname[:28], mode="lines", line=dict(color=palette[i_ % len(palette)], width=3)))
            style_fig(fig, 360)
            fig.update_layout(xaxis_title="Hour of day (0 to 23)", yaxis_title="Average demand (kW)")
            show(fig, "cmp_profile")
        with cc2:
            chart_head("Monthly energy per home", "Average kWh per month of data.", "🔋")
            fig = go.Figure(go.Bar(x=[h_[:22] for h_ in cmp_df["Home"]], y=cmp_df["Monthly energy (kWh)"], name="Monthly kWh",
                                   marker_color=[palette[i_ % len(palette)] for i_ in range(len(cmp_df))]))
            style_fig(fig, 360, "x")
            fig.update_layout(yaxis_title="Energy (kWh)", xaxis_title="Home")
            show(fig, "cmp_bar")
        st.download_button("⬇ Download comparison (CSV)", cmp_df.to_csv(index=False).encode("utf-8"), "home_comparison.csv", "text/csv", key="dl_cmp")
        st.caption("Homes may cover different months, so compare patterns and ratios more than raw totals.")

# ============================================================
# TAB 9 - GUIDE (NEW)
# ============================================================

with tab_guide:
    g1, g2 = st.columns([1, 2])
    with g1:
        st.markdown(f'<div class="guide-art">{HOUSE_SVG}<div style="font-weight:700;margin-top:6px">Your smart home, explained</div></div>', unsafe_allow_html=True)
    with g2:
        st.markdown(f"### 📘 Welcome to {APP_NAME}")
        st.markdown(
            "This app learns how your home uses electricity, predicts what comes next and warns you when something looks unusual. "
            "You do not need any technical background: every chart has a short explanation above it."
        )
        st.markdown('<span class="badge">Forecasting</span> <span class="badge">Anomaly detection</span> '
                    '<span class="badge">Savings</span> <span class="badge">Your own data</span>', unsafe_allow_html=True)

    st.markdown("### 🚀 Get started in four steps")
    gs = st.columns(4)
    tip(gs[0], "1️⃣", "Load your data", "Use the built-in data or upload a CSV/Excel file in the sidebar. Download the template if you are unsure of the format.", TEAL)
    tip(gs[1], "2️⃣", "Read the overview", "Check current demand, the next-hour forecast, your efficiency score and how you compare with earlier periods.", VIOLET)
    tip(gs[2], "3️⃣", "Review alerts", "Open Anomalies to set your own limit and see which readings were unusual and how severe they were.", RED)
    tip(gs[3], "4️⃣", "Plan and save", "Use Savings for bill estimates, budgets and appliance costs, and Forecast lab to plan the next 24 hours.", AMBER)

    st.markdown("### 🗂️ What each tab does")
    st.markdown(
        "- **🏠 Overview**: live monitor, next-hour forecast, efficiency score, insights and period comparison.\n"
        "- **📊 Usage patterns**: daily, hourly, weekly and monthly patterns, load duration curve and appliance breakdown.\n"
        "- **🚨 Anomalies**: unusual readings, alert limit, severity ranking and CSV export.\n"
        "- **💰 Savings**: bill estimates, carbon estimate, monthly budget planner and appliance cost calculator.\n"
        "- **🤖 AI assistant**: ask questions in plain English and export the chat or a report.\n"
        "- **🔎 Explorer**: any date range, any resolution, data health check and CSV download.\n"
        "- **🔮 Forecast lab**: up to 24 hours ahead with a likely-range band, a baseline for comparison and a model performance backtest (MAE, RMSE, MAPE).\n"
        "- **🏘️ Compare homes**: upload several meter files and compare usage, peaks and base load side by side.\n"
        "- **Extras**: time-of-use tariff planner, peak-shift simulator, weekly saving goals with streaks, daily digest, PDF report, language switch (English, Sinhala, Tamil) and dark mode in the sidebar Settings."
    )

    st.markdown("### 📖 Simple glossary")
    with st.expander("kW vs kWh: what is the difference?"):
        st.write("kW (kilowatt) is how much power your home draws right now, like the speed of a car. "
                 "kWh (kilowatt-hour) is the total energy used over time, like the distance travelled. Your bill is based on kWh.")
    with st.expander("What is an anomaly?"):
        st.write("A reading that is very different from what is normal for that hour of the day. It might be a party, a faulty appliance, "
                 "a device left on, or simply a holiday. It does not automatically mean something is broken.")
    with st.expander("What is base load?"):
        st.write("The electricity your home uses even when nobody is active, such as the fridge, router and standby devices. "
                 "A high overnight base load is a common source of wasted energy.")
    with st.expander("What is load factor?"):
        st.write("Average demand divided by peak demand. A higher load factor means steady use. A low one means short, sharp peaks that are usually more expensive to supply.")
    with st.expander("How is the efficiency score calculated?"):
        st.write("It starts at 100 and loses points for sharp peaks compared with your 24-hour average, high overnight use and recent anomalies. "
                 "It is a guide to help you compare over time, not an official rating.")

    st.markdown("### ❓ Frequently asked questions")
    with st.expander("What file format can I upload?"):
        st.write("CSV, Excel (.xlsx/.xls), TXT or TSV with one date/time column and one power or energy column. "
                 "Choose the correct unit (kW, W, Wh or kWh per hour). Readings are converted to hourly values automatically and small gaps are filled.")
    with st.expander("Why is the AI service offline?"):
        st.write(f"The forecasts and bill estimates come from a separate FastAPI service at {API_URL}. "
                 "Start it, then press **Refresh data and AI services** in the sidebar. Charts, anomalies and the baseline forecast still work without it.")
    with st.expander("How many rows do I need for forecasting?"):
        st.write("At least 169 hourly rows: 7 days of history plus 1 hour, because the model uses values from the same hour last week.")
    with st.expander("Is this live data?"):
        st.write("No. It replays recorded readings. The 'Live playback' control on the Overview tab only animates the history so you can see how patterns evolve.")
    with st.expander("Can I trust the bill estimate?"):
        st.write("It is an estimate from the cost API. Real bills depend on your tariff, customer category, fixed charges and taxes.")

        st.write("Yes. Layouts stack into one or two columns, tabs swipe sideways, and inputs and buttons are larger for touch. "
                 "Open **Settings** in the sidebar (tap the arrow at the top left) to switch on mobile-friendly charts, dark mode or another language. "
                 "Phone browsers are detected automatically where possible.")
    with st.expander("How is the forecast range calculated?"):
        st.write("Run a backtest in the Forecast lab. The typical size of the model's recent mistakes sets the width of the band, which grows with each extra hour ahead. "
                 "It is an approximation, so real demand can fall outside it.")
    with st.expander("Are the Sinhala and Tamil translations complete?"):
        st.write("The main tab names, headings and key labels are translated. Detailed explanations stay in English for now.")
    st.info("This project is a portfolio and academic demonstration. It is not a certified metering or billing tool.")

st.markdown(
    f'<div class="foot"><b>© 2026 Dumindu Wickramaarachchi</b><br>{APP_NAME} · Portfolio / Academic ML Project</div>',
    unsafe_allow_html=True,
)