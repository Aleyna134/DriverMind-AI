import streamlit as st

from styles.theme import GOOD, WARNING, TEXT_SECONDARY
from components.risk_factors import factor_color, GOOD_FACTORS

ICON_SHIELD = """<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"
stroke-linecap="round" stroke-linejoin="round" width="34" height="34">
<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path>
<polyline points="9 12 11 14 15 10"></polyline>
</svg>"""

ICON_ALERT = """<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"
stroke-linecap="round" stroke-linejoin="round" width="34" height="34">
<path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path>
<line x1="12" y1="9" x2="12" y2="13"></line>
<line x1="12" y1="17" x2="12.01" y2="17"></line>
</svg>"""

ICON_MOON = """<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"
stroke-linecap="round" stroke-linejoin="round" width="34" height="34">
<path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path>
</svg>"""

# factor -> (icon, title, message)
RECOMMENDATIONS = {
    "Henüz tahmin yapılmadı": (
        ICON_SHIELD,
        "Ready to monitor",
        'Press "Start" to begin live risk monitoring.',
    ),
    "No Critical Risk": (
        ICON_SHIELD,
        "Good driving!",
        "You are driving safely. Maintain your current behaviour to keep the risk low.",
    ),
    "Sürekli güvenli sürüş": (
        ICON_SHIELD,
        "Excellent driving!",
        "You've kept a safe, steady behaviour for a while now. Keep it up.",
    ),
    "High Speed": (
        ICON_ALERT,
        "Reduce your speed",
        "You are driving faster than a safe range. Ease off the accelerator.",
    ),
    "Short Following Distance": (
        ICON_ALERT,
        "Increase following distance",
        "You're following too closely. Leave more space to react safely.",
    ),
    "Lane Departure": (
        ICON_ALERT,
        "Return to your lane",
        "You've drifted from your lane. Correct your position smoothly.",
    ),
    "High Closing Speed": (
        ICON_ALERT,
        "Slow down — closing fast",
        "You're approaching the vehicle ahead quickly. Brake early and smoothly.",
    ),
    "Harsh Braking": (
        ICON_ALERT,
        "Avoid harsh braking",
        "Sudden braking increases risk. Anticipate stops and decelerate gradually.",
    ),
    "Harsh Acceleration": (
        ICON_ALERT,
        "Ease off the accelerator",
        "Rapid acceleration was detected. Accelerate smoothly instead.",
    ),
    "Sharp Steering": (
        ICON_ALERT,
        "Avoid sudden steering",
        "A sharp steering movement was detected. Steer gently and stay centered.",
    ),
    "Lane Drift": (
        ICON_MOON,
        "Stay alert",
        "Your lane position is drifting steadily. Consider taking a break.",
    ),
}

DEFAULT_KEY = "No Critical Risk"


def render_recommendation(risk_factors):

    factor = risk_factors[0] if risk_factors else DEFAULT_KEY
    icon, title, message = RECOMMENDATIONS.get(factor, RECOMMENDATIONS[DEFAULT_KEY])

    if factor in GOOD_FACTORS:
        color = GOOD if factor != "Henüz tahmin yapılmadı" else TEXT_SECONDARY
    else:
        color = factor_color(factor) if factor != "Lane Drift" else WARNING

    st.markdown(
        '<div class="section-heading">Driving recommendation</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<div style="display:flex; align-items:center; gap:26px;">'
        f'<div style="width:76px; height:76px; min-width:76px; border-radius:50%; background:{color}1f; border:1px solid {color}55; display:flex; align-items:center; justify-content:center; color:{color};">{icon}</div>'
        f'<div>'
        f'<div style="font-family:\'Space Grotesk\',system-ui,sans-serif; font-size:1.4rem; font-weight:700; color:{color}; margin-bottom:6px;">{title}</div>'
        f'<div style="color:{TEXT_SECONDARY}; font-size:1.05rem; line-height:1.5;">{message}</div>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
