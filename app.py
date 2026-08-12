"""SmartCharge RL application router with Smart Recommendation as the landing page."""
import streamlit as st

st.set_page_config(page_title="SmartCharge RL", layout="wide", initial_sidebar_state="expanded")

navigation = st.navigation([
    st.Page("pages/1_Smart_Recommendation.py", title="Smart Recommendation", default=True),
    st.Page("pages/2_Network_Monitor.py", title="Network Monitor"),
    st.Page("pages/3_Model_Analysis.py", title="Model Analysis"),
    st.Page("pages/4_RL_Simulation.py", title="RL Simulation"),
    st.Page("pages/5_Methodology.py", title="Methodology"),
])

if "ui_theme" not in st.session_state:
    st.session_state.ui_theme = "dark"

with st.sidebar:
    st.markdown("## ⚡ SmartCharge RL")
    st.caption("Intelligent EV route and charging planner")
    st.page_link("pages/1_Smart_Recommendation.py", label="Plan a Charge", icon=":material/ev_station:")

if st.session_state.ui_theme == "light":
    theme = {
        "background": "#f5f7fa", "surface": "#ffffff", "panel": "rgba(255,255,255,.94)",
        "metric": "#f3f6f9", "text": "#14202b", "muted": "#607080", "line": "#dbe3ea",
        "accent": "#16a36f", "accent_soft": "rgba(22,163,111,.08)", "sidebar": "#ffffff",
    }
else:
    theme = {
        "background": "#0b1017", "surface": "#101722", "panel": "rgba(15,23,42,.72)",
        "metric": "rgba(30,41,59,.78)", "text": "#f1f5f9", "muted": "#94a3b8", "line": "#263448",
        "accent": "#18c7b5", "accent_soft": "rgba(24,199,181,.08)", "sidebar": "#111824",
    }

st.markdown(f"""
<style>
  :root {{
    --app-bg:{theme['background']}; --surface:{theme['surface']}; --panel:{theme['panel']};
    --metric-bg:{theme['metric']}; --text:{theme['text']}; --muted:{theme['muted']};
    --line:{theme['line']}; --accent:{theme['accent']}; --accent-soft:{theme['accent_soft']};
  }}
  .stApp {{background:{theme['background']}; color:{theme['text']};}}
  .stApp p, .stApp label, .stApp h1, .stApp h2, .stApp h3,
  [data-testid="stSidebar"] span, [data-testid="stSidebar"] p {{color:{theme['text']};}}
  .stApp .muted, .stApp .tagline, .stApp .metric-label, .stApp .data-note {{color:{theme['muted']} !important;}}
  [data-testid="stHeader"] {{background:transparent; height:2rem;}}
  [data-testid="stToolbar"] {{top:.2rem;}}
  [data-testid="stSidebar"] {{background:{theme['sidebar']}; border-right:1px solid {theme['line']};}}
  [data-testid="stSidebarNav"] {{display:none !important;}}
  [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2 {{margin-top:.2rem; color:{theme['text']};}}
  [data-testid="stSidebar"] a {{border-radius:8px;}}
  [data-baseweb="input"], [data-baseweb="select"] > div,
  [data-testid="stNumberInput"] input, [data-testid="stTextInput"] input {{
    background:{theme['surface']} !important; color:{theme['text']} !important; border-color:{theme['line']} !important;
  }}
  [data-testid="stSelectbox"] [role="combobox"],
  [data-testid="stNumberInput"] > div > div {{
    background:{theme['surface']} !important; color:{theme['text']} !important; border-color:{theme['line']} !important;
  }}
  [data-testid="stSelectbox"] [role="group"], [data-testid="stSelectbox"] button,
  [data-testid="stNumberInput"] [role="group"], [data-testid="stNumberInput"] button {{
    background:{theme['surface']} !important; color:{theme['text']} !important; border-color:{theme['line']} !important;
  }}
  [data-testid="stSelectbox"] svg, [data-testid="stNumberInput"] svg {{fill:{theme['text']} !important;}}
  [data-baseweb="select"] span, [data-testid="stNumberInput"] button,
  [data-testid="stTextInput"] input::placeholder {{color:{theme['muted']} !important;}}
  button[kind="secondary"] {{background:{theme['surface']} !important; color:{theme['text']} !important; border-color:{theme['line']} !important;}}
  [data-testid="stExpander"] {{background:{theme['panel']}; border-color:{theme['line']};}}
  [data-testid="stSlider"] [role="slider"] {{background:{theme['accent']} !important; border-color:{theme['accent']} !important;}}
  [data-testid="stFormSubmitButton"] button {{background:{theme['accent']} !important; color:white !important; border:0 !important;}}
  [data-testid="stMetricLabel"] p {{color:{theme['muted']} !important;}}
  [data-testid="stMetricValue"] {{color:{theme['text']} !important;}}
  .block-container {{padding-top:3.25rem !important;}}
</style>
""", unsafe_allow_html=True)

navigation.run()
