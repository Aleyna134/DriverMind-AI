import altair as alt
import pandas as pd
import streamlit as st

from styles.theme import BLUE, CYAN, VIOLET, WARNING, TEXT_SECONDARY

# (feature key, label, color, index into a FEATURE_COLUMNS row, value format)
SENSORS = [
    ("speed", "Speed (km/h)", BLUE, 6, "{:.0f}"),
    ("front_distance", "Front Distance (m)", CYAN, 10, "{:.2f}"),
    ("lane_offset", "Lane Offset (m)", VIOLET, 8, "{:.2f}"),
    ("relative_speed", "Relative Speed (m/s)", WARNING, 11, "{:.2f}"),
]

SPARKLINE_WINDOW = 40


def _sparkline(values, color):
    data = pd.DataFrame({"tick": range(len(values)), "value": values})

    return (
        alt.Chart(data)
        .mark_line(color=color, strokeWidth=2.5, interpolate="monotone")
        .encode(
            x=alt.X("tick:Q", axis=None),
            y=alt.Y("value:Q", axis=None, scale=alt.Scale(zero=False, nice=False)),
        )
        .properties(height=64, background="transparent")
        .configure_view(strokeWidth=0)
    )


def render_sensor_summary(feature_history):

    st.markdown(
        '<div class="section-heading">Live Sensor Overview</div>',
        unsafe_allow_html=True,
    )

    cols = st.columns(4, gap="medium")

    for col, (key, label, color, idx, fmt) in zip(cols, SENSORS):
        with col:
            with st.container(border=True):
                if feature_history:
                    values = [float(row[idx]) for row in feature_history[-SPARKLINE_WINDOW:]]
                else:
                    values = [0.0]
                current = values[-1]
                if len(values) < 2:
                    values = values * 2

                st.markdown(
                    f'<div style="color:{TEXT_SECONDARY}; font-size:0.92rem; font-weight:600;">{label}</div>'
                    f'<div style="color:{color}; font-size:1.6rem; font-weight:800; '
                    f'font-family:\'Space Grotesk\',system-ui,sans-serif; margin-bottom:2px;">{fmt.format(current)}</div>',
                    unsafe_allow_html=True,
                )
                st.altair_chart(_sparkline(values, color), width="stretch")
