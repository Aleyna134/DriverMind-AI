import altair as alt
import pandas as pd
import streamlit as st

from styles.theme import CYAN, VIOLET, TEXT_SECONDARY, BORDER


def render_trend_chart(risk_history):

    st.markdown(
        '<div class="section-heading">Risk trend</div>',
        unsafe_allow_html=True,
    )

    if not risk_history:
        empty_style = f'color:{TEXT_SECONDARY}; font-size:1rem; padding:2.5rem 1rem; text-align:center; border:1px dashed {BORDER}; border-radius:12px;'
        st.markdown(
            f'<div style="{empty_style}">No predictions yet — press "Start" to begin the live stream.</div>',
            unsafe_allow_html=True,
        )
        return

    data = pd.DataFrame({
        "tick": range(len(risk_history)),
        "risk": risk_history,
    })

    hover = alt.selection_point(
        fields=["tick"],
        nearest=True,
        on="mouseover",
        empty=False,
    )

    base = alt.Chart(data).encode(
        x=alt.X(
            "tick:Q",
            axis=alt.Axis(
                title=None,
                grid=False,
                domainColor=BORDER,
                tickColor=BORDER,
                labelColor=TEXT_SECONDARY,
                labelFontSize=12,
            ),
        ),
        y=alt.Y(
            "risk:Q",
            scale=alt.Scale(domain=[0, 100]),
            axis=alt.Axis(
                title=None,
                gridColor=BORDER,
                gridDash=[2, 3],
                domain=False,
                tickColor=BORDER,
                labelColor=TEXT_SECONDARY,
                labelFontSize=12,
            ),
        ),
    )

    area = base.mark_area(
        line=False,
        color=alt.Gradient(
            gradient="linear",
            stops=[
                alt.GradientStop(color=CYAN, offset=0),
                alt.GradientStop(color="transparent", offset=1),
            ],
            x1=1, x2=1, y1=1, y2=0,
        ),
        opacity=0.25,
    )

    line = base.mark_line(
        color=alt.Gradient(
            gradient="linear",
            stops=[
                alt.GradientStop(color=VIOLET, offset=0),
                alt.GradientStop(color=CYAN, offset=1),
            ],
            x1=0, x2=1, y1=0, y2=0,
        ),
        strokeWidth=3,
        interpolate="monotone",
    )

    points = base.mark_circle(size=70, color=CYAN, opacity=0).encode(
        opacity=alt.condition(hover, alt.value(1), alt.value(0)),
        tooltip=[
            alt.Tooltip("tick:Q", title="Step"),
            alt.Tooltip("risk:Q", title="Risk score"),
        ],
    ).add_params(hover)

    rule = base.mark_rule(color=BORDER).encode(
        opacity=alt.condition(hover, alt.value(1), alt.value(0)),
    )

    chart = (area + line + rule + points).properties(
        height=320,
        background="transparent",
    ).configure_view(strokeWidth=0)

    st.altair_chart(chart, width="stretch")
