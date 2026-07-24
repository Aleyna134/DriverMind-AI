import streamlit as st


def _step(key, delta, lo, hi, decimals):
    new_value = st.session_state[key] + delta
    new_value = max(lo, min(hi, new_value))
    st.session_state[key] = round(new_value, decimals) if decimals else int(new_value)


def stepped_slider(label, key, min_value, max_value, value, step, decimals=0):

    if key not in st.session_state:
        st.session_state[key] = value

    minus_col, slider_col, plus_col = st.columns([1, 9, 1], vertical_alignment="center")

    with minus_col:
        st.button(
            "−",
            key=f"{key}_minus",
            width="stretch",
            on_click=_step,
            args=(key, -step, min_value, max_value, decimals),
        )

    with slider_col:
        result = st.slider(
            label,
            min_value=min_value,
            max_value=max_value,
            step=step,
            key=key,
            label_visibility="visible",
        )

    with plus_col:
        st.button(
            "+",
            key=f"{key}_plus",
            width="stretch",
            on_click=_step,
            args=(key, step, min_value, max_value, decimals),
        )

    return result


def render_feature_controls():

    st.markdown(
        '<div class="section-heading">Input features</div>',
        unsafe_allow_html=True,
    )

    # ---------------- Main Features ----------------

    speed = stepped_slider(
        "Speed (km/h)",
        "speed",
        min_value=0,
        max_value=180,
        value=80,
        step=1,
    )

    front_distance = stepped_slider(
        "Front distance (m)",
        "front_distance",
        min_value=0.0,
        max_value=220.0,
        value=25.0,
        step=0.5,
        decimals=1,
    )

    lane_offset = stepped_slider(
        "Lane offset (m)",
        "lane_offset",
        min_value=-2.0,
        max_value=2.0,
        value=0.0,
        step=0.01,
        decimals=2,
    )

    relative_speed = stepped_slider(
        "Relative speed (m/s)",
        "relative_speed",
        min_value=-1.0,
        max_value=13.0,
        value=0.0,
        step=0.1,
        decimals=1,
    )

    st.divider()

    start_col, toggle_col = st.columns(2)

    is_running = bool(st.session_state.get("running"))
    toggle_label = "Stop" if is_running else "Continue"

    with start_col:
        start = st.button(
            "Start",
            width="stretch",
            type="primary",
        )

    with toggle_col:
        toggle = st.button(
            toggle_label,
            width="stretch",
        )

    return {
        "speed": float(speed),
        "front_distance": float(front_distance),
        "lane_offset": float(lane_offset),
        "relative_speed": float(relative_speed),
        "start": start,
        "toggle": toggle,
    }
