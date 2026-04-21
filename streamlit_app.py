"""
Streamlit Dashboard for DeepAR + N-BEATS Engine.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from huggingface_hub import HfApi, hf_hub_download
import json
import config
from us_calendar import USMarketCalendar

st.set_page_config(page_title="P2Quant DeepAR + N-BEATS", page_icon="📈", layout="wide")

st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: 600; color: #1f77b4; }
    .hero-card { background: linear-gradient(135deg, #1f77b4 0%, #2C5282 100%); border-radius: 16px; padding: 2rem; color: white; }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def load_latest_results():
    try:
        api = HfApi(token=config.HF_TOKEN)
        files = api.list_repo_files(repo_id=config.HF_OUTPUT_REPO, repo_type="dataset")
        json_files = sorted([f for f in files if f.endswith('.json')], reverse=True)
        if not json_files:
            return None
        local_path = hf_hub_download(
            repo_id=config.HF_OUTPUT_REPO, filename=json_files[0],
            repo_type="dataset", token=config.HF_TOKEN, cache_dir="./hf_cache"
        )
        with open(local_path) as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return None

def display_model_tab(model_data: dict, model_name: str):
    top_picks = model_data['top_picks']
    universes_data = model_data['universes']

    subtabs = st.tabs(["📊 Combined", "📈 Equity Sectors", "💰 FI/Commodities"])
    universe_keys = ["COMBINED", "EQUITY_SECTORS", "FI_COMMODITIES"]

    for subtab, key in zip(subtabs, universe_keys):
        with subtab:
            if key in universes_data:
                universe_dict = universes_data[key]
                picks = top_picks.get(key, [])
                if picks:
                    top = picks[0]
                    st.markdown(f"""
                    <div class="hero-card">
                        <h2>🏆 {model_name} Top Pick: {top['ticker']}</h2>
                        <p>1d Forecast: {top.get('forecast_1d', 0)*100:.3f}%</p>
                        <p>5d: {top.get('forecast_5d', 0)*100:.3f}% | 22d: {top.get('forecast_22d', 0)*100:.3f}%</p>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("### All ETFs")
                rows = []
                for t, d in universe_dict.items():
                    rows.append({
                        'Ticker': t,
                        '1d': f"{d.get('forecast_1d', 0)*100:.3f}%",
                        '5d': f"{d.get('forecast_5d', 0)*100:.3f}%",
                        '22d': f"{d.get('forecast_22d', 0)*100:.3f}%"
                    })
                df = pd.DataFrame(rows).sort_values('1d', ascending=False)
                st.dataframe(df, use_container_width=True, hide_index=True)

# --- Sidebar ---
st.sidebar.markdown("## ⚙️ Configuration")
calendar = USMarketCalendar()
st.sidebar.markdown(f"**📅 Next Trading Day:** {calendar.next_trading_day().strftime('%Y-%m-%d')}")
data = load_latest_results()
if data:
    st.sidebar.markdown(f"**Run Date:** {data.get('run_date', 'Unknown')}")

st.markdown('<div class="main-header">📈 P2Quant DeepAR + N‑BEATS</div>', unsafe_allow_html=True)
st.markdown('<div>Multi‑Horizon Forecasting with Two State‑of‑the‑Art Models</div>', unsafe_allow_html=True)

if data is None:
    st.warning("No data available.")
    st.stop()

# --- Main Tabs: DeepAR and N-BEATS ---
main_tab1, main_tab2 = st.tabs(["🧠 DeepAR", "⚡ N‑BEATS"])

with main_tab1:
    if 'deepar' in data:
        display_model_tab(data['deepar'], "DeepAR")
    else:
        st.warning("DeepAR data not available.")

with main_tab2:
    if 'nbeats' in data:
        display_model_tab(data['nbeats'], "N‑BEATS")
    else:
        st.warning("N‑BEATS data not available.")
