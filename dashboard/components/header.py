import base64
from pathlib import Path

import streamlit as st

LOGO_PATH = Path(__file__).resolve().parent.parent / "assets" / "logo.png"
LOGO_B64 = base64.b64encode(LOGO_PATH.read_bytes()).decode()


def render_header():

    st.markdown(
        f'<div style="display:flex; align-items:center; gap:6px; flex-wrap:wrap;">'
        f'<img src="data:image/png;base64,{LOGO_B64}" style="width:96px; height:96px;" />'
        f'<div class="hero-title">DriverMind AI</div>'
        f'<div class="hero-subtitle" style="margin-top:0; margin-left:90px; align-self:center;">Real-time Driver Behaviour Risk Monitoring</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
