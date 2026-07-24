import streamlit as st

from styles.theme import GOOD, WARNING, CRITICAL

GOOD_FACTORS = {
    "No Critical Risk",
    "Henüz tahmin yapılmadı",
    "Sürekli güvenli sürüş",
}


def factor_color(factor):
    if factor in GOOD_FACTORS:
        return GOOD

    if factor == "Lane Drift":
        return WARNING

    return CRITICAL


def render_risk_factors(factors):

    st.markdown(
        '<div class="section-heading">Risk factors</div>',
        unsafe_allow_html=True,
    )

    chips = "".join(
        f'<span class="chip" style="background:{factor_color(factor)}1f; color:{factor_color(factor)}; border-color:{factor_color(factor)}55;"><span class="chip-dot" style="background:{factor_color(factor)};"></span>{factor}</span>'
        for factor in factors
    )

    st.markdown(
        f'<div class="chip-row">{chips}</div>',
        unsafe_allow_html=True,
    )
