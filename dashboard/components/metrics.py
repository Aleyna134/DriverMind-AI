import streamlit as st

from styles.theme import (
    SURFACE_ALT,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    STATUS_BY_CLASS,
    VIOLET,
    risk_color,
)


def stat_card(label, value, accent, sub=None, big=False):
    sub_html = (
        f'<span style="color:{TEXT_SECONDARY}; font-size:0.95rem; font-weight:600; white-space:nowrap;">{sub}</span>'
        if sub
        else ""
    )

    value_size = "3.2rem" if big else "2.1rem"

    card_style = f"background: linear-gradient(160deg, {accent}22 0%, {SURFACE_ALT} 55%); border: 1px solid {accent}55; box-shadow: 0 10px 30px {accent}1f;"
    label_style = f"color:{TEXT_SECONDARY};"
    value_style = f"color:{TEXT_PRIMARY}; font-size:{value_size}; white-space:nowrap;"

    st.markdown(
        f'<div class="stat-card" style="{card_style}"><div class="stat-card-label" style="{label_style}">{label}</div><div style="display:flex; align-items:baseline; gap:10px; margin-top:0.15rem;"><div class="stat-card-value" style="{value_style}">{value}</div>{sub_html}</div></div>',
        unsafe_allow_html=True,
    )


def render_metrics(prediction):

    col1, col2, col3 = st.columns(3)

    risk_score = prediction["risk_score"]
    prediction_class = prediction["prediction"]
    confidence = prediction["confidence"]

    with col1:
        stat_card(
            "Risk score",
            risk_score,
            risk_color(risk_score),
            sub="out of 100",
            big=True,
        )

    with col2:
        stat_card(
            "Prediction",
            prediction_class,
            STATUS_BY_CLASS.get(prediction_class, TEXT_SECONDARY),
        )

    with col3:
        stat_card(
            "Model confidence",
            f"{confidence:.1f}%",
            VIOLET,
        )
