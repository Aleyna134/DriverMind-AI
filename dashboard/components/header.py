import streamlit as st

from styles.theme import GOOD, ACCENT


def render_header():

    title_col, mode_col = st.columns([3, 1])

    with title_col:
        st.markdown(
            '<div class="hero-title">Driver Risk Intelligence</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="hero-subtitle">Real-time driver behaviour risk monitoring</div>',
            unsafe_allow_html=True,
        )

    with mode_col:
        mode = st.toggle("Simulation Mode", value=True)

        dot_color = ACCENT if mode else GOOD
        label = "Simulation" if mode else "Live"

        st.markdown(
            f'<div class="status-pill" style="margin-top:4px;"><span class="status-dot" style="background:{dot_color};"></span>{label} mode active</div>',
            unsafe_allow_html=True,
        )

    return mode
