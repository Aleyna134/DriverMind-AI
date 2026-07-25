"""
theme.py

Shared design tokens (colors, fonts) and global CSS for the dashboard.
Widget colors (sliders, buttons, toggles) come from the native Streamlit
theme in .streamlit/config.toml — this file adds fonts, card chrome,
and the custom chip/pill/hero components.
"""

import streamlit as st

# ---------------- Palette ----------------
# Kept in sync with .streamlit/config.toml.

BG_GRADIENT = "linear-gradient(160deg, #060a16 0%, #0b1226 45%, #10152b 100%)"

SURFACE = "#101a30"
SURFACE_ALT = "#0d1628"
BORDER = "rgba(255, 255, 255, 0.09)"
BORDER_STRONG = "rgba(255, 255, 255, 0.18)"

TEXT_PRIMARY = "#f5f7fc"
TEXT_SECONDARY = "#a3b1c9"
TEXT_MUTED = "#66759190"

# Vivid multi-hue accent system.
BLUE = "#5b8cff"
VIOLET = "#a463f7"
CYAN = "#26e0d6"
PINK = "#ff5ec4"

ACCENT = BLUE
ACCENT_SOFT = "rgba(91, 140, 255, 0.16)"

GRADIENT_HERO = f"linear-gradient(135deg, {BLUE} 0%, {VIOLET} 55%, {PINK} 100%)"

GOOD = "#2ee672"
WARNING = "#ffc93c"
SERIOUS = "#ff8a4c"
CRITICAL = "#ff4d6a"

STATUS_BY_CLASS = {
    "Normal": GOOD,
    "Drowsy": WARNING,
    "Aggressive": CRITICAL,
    "-": TEXT_SECONDARY,
}


def risk_color(risk_score):
    if risk_score < 30:
        return GOOD
    if risk_score < 60:
        return WARNING
    if risk_score < 80:
        return SERIOUS
    return CRITICAL


def inject_theme():
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700;800&family=Inter:wght@400;500;600;700;800&family=Dancing+Script:wght@500;600;700&display=swap');

        html, body {{
            font-family: 'Inter', system-ui, -apple-system, sans-serif;
            font-size: 17px;
        }}

        [data-testid="stMarkdownContainer"] p,
        [data-testid="stWidgetLabel"] p {{
            font-family: 'Inter', system-ui, -apple-system, sans-serif;
            font-size: 1rem;
        }}

        h1, h2, h3, h4, h5,
        .display-font {{
            font-family: 'Space Grotesk', system-ui, -apple-system, sans-serif;
        }}

        #MainMenu, footer {{ visibility: hidden; }}

        [data-testid="stHeader"] {{
            height: 0;
            min-height: 0;
            visibility: hidden;
        }}

        .stApp {{
            background: {BG_GRADIENT};
        }}

        [data-testid="stMainBlockContainer"],
        [data-testid="block-container"] {{
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: min(1600px, 94vw);
            margin: 0 auto;
        }}

        /* Breathing room between stacked widgets (sliders, buttons, etc.) */
        [data-testid="stVerticalBlock"] {{
            gap: 1.15rem;
        }}

        /* ---------------- Cards ---------------- */
        [data-testid="stVerticalBlockBorderWrapper"] {{
            background: {SURFACE};
            border-radius: 20px;
        }}

        [data-testid="stVerticalBlockBorderWrapper"] > div > div {{
            padding: 0.4rem 0.5rem;
        }}

        /* Left "Input features" panel: match the right panel's height and
        vertically center its content instead of leaving it stuck to the top.
        The column is a flex row item (stretched to the tallest sibling by
        default) — this propagates that stretch down through the wrapper
        chain so the bordered box itself actually fills it, then centers
        its contents inside the extra space. */
        div[data-testid="stColumn"]:has(.st-key-input_features_panel) {{
            display: flex;
            flex-direction: column;
        }}

        div[data-testid="stColumn"]:has(.st-key-input_features_panel) > div[data-testid="stVerticalBlock"],
        div[data-testid="stColumn"]:has(.st-key-input_features_panel) [data-testid="stLayoutWrapper"] {{
            flex: 1;
            display: flex;
            flex-direction: column;
        }}

        .st-key-input_features_panel {{
            flex: 1;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }}

        /* Nudge the stepper "+" buttons closer to their sliders (only the
        plus side — the minus side gap stays as-is). */
        div[class*="st-key-"][class*="_plus"] {{
            margin-left: -14px;
        }}

        /* "Driving Recommendation" panel: taller, with extra bottom padding
        so the icon/text row doesn't sit flush against the card's edge —
        same padding scale as the stat cards elsewhere on the dashboard. */
        .st-key-recommendation_panel {{
            min-height: 190px;
            padding: 1.5rem 1.5rem 2.25rem;
            box-sizing: border-box;
        }}

        /* Right "monitoring" panel: the risk-factor chips at the bottom
        were sitting flush against the card edge — give it the same
        extra bottom breathing room as the recommendation panel. */
        .st-key-monitoring_panel {{
            padding-bottom: 2.25rem;
            box-sizing: border-box;
        }}

        /* ---------------- Headings ---------------- */
        h1, h2, h3, h4 {{
            font-weight: 700;
            letter-spacing: -0.02em;
            color: {TEXT_PRIMARY};
        }}

        .section-heading {{
            font-family: 'Space Grotesk', system-ui, sans-serif;
            font-size: 1.35rem;
            font-weight: 700;
            color: {TEXT_PRIMARY};
            margin-bottom: 0.7rem;
            letter-spacing: -0.01em;
        }}

        .hero-title {{
            font-family: 'Space Grotesk', system-ui, sans-serif;
            font-size: 3.1rem;
            font-weight: 800;
            letter-spacing: -0.03em;
            line-height: 1.05;
            background: {GRADIENT_HERO};
            -webkit-background-clip: text;
            background-clip: text;
            -webkit-text-fill-color: transparent;
            filter: drop-shadow(0 0 26px rgba(91, 140, 255, 0.35));
        }}

        .hero-subtitle {{
            font-family: 'Dancing Script', cursive;
            color: {TEXT_SECONDARY};
            font-size: 1.9rem;
            margin-top: 0.35rem;
            font-weight: 600;
            letter-spacing: 0.01em;
        }}

        /* ---------------- Buttons ---------------- */
        .stButton > button {{
            border-radius: 12px;
            font-weight: 700;
            font-size: 1.02rem;
            padding: 0.7rem 1.1rem;
            display: flex;
            align-items: center;
            justify-content: center;
            line-height: 1;
        }}

        .stButton > button[kind="primary"] {{
            background: {GRADIENT_HERO};
            border: none;
            box-shadow: 0 8px 24px rgba(164, 99, 247, 0.35);
        }}

        .stButton > button[kind="primary"]:hover {{
            filter: brightness(1.08);
            box-shadow: 0 10px 30px rgba(164, 99, 247, 0.5);
        }}

        [data-testid="stWidgetLabel"] p {{
            font-weight: 600;
            font-size: 1.02rem;
        }}

        /* ---------------- Metric widget ---------------- */
        [data-testid="stMetric"] {{
            background: {SURFACE_ALT};
            border: 1px solid {BORDER};
            border-radius: 16px;
            padding: 1rem 1.2rem;
        }}

        [data-testid="stMetricLabel"] p {{
            color: {TEXT_SECONDARY};
            font-size: 1rem;
            font-weight: 600;
        }}

        [data-testid="stMetricValue"] {{
            font-size: 2rem;
        }}

        /* ---------------- Badges / chips ---------------- */
        .chip-row {{
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
        }}

        .chip {{
            display: inline-flex;
            align-items: center;
            gap: 7px;
            padding: 9px 18px;
            border-radius: 999px;
            font-size: 0.98rem;
            font-weight: 700;
            border: 1px solid transparent;
        }}

        .chip-dot {{
            width: 8px;
            height: 8px;
            border-radius: 999px;
            flex-shrink: 0;
        }}

        .status-pill {{
            display: inline-flex;
            align-items: center;
            gap: 9px;
            padding: 9px 18px;
            border-radius: 999px;
            font-weight: 700;
            font-size: 1rem;
            border: 1px solid {BORDER};
            background: {SURFACE_ALT};
            color: {TEXT_SECONDARY};
        }}

        .status-dot {{
            width: 9px;
            height: 9px;
            border-radius: 999px;
        }}

        /* ---------------- Stat cards ---------------- */
        .stat-card {{
            border-radius: 18px;
            padding: 1.3rem 1.5rem;
            position: relative;
            overflow: hidden;
            min-height: 165px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            box-sizing: border-box;
        }}

        /* ---------------- Timer ---------------- */
        .timer-box {{
            background: {SURFACE_ALT};
            border: 1px solid {BORDER};
            border-radius: 14px;
            padding: 0.5rem 1rem;
            text-align: center;
            font-family: 'Space Grotesk', system-ui, sans-serif;
            font-size: 1.9rem;
            font-weight: 800;
            letter-spacing: 0.01em;
            font-variant-numeric: tabular-nums;
            color: {TEXT_PRIMARY};
        }}

        .stat-card-label {{
            font-size: 0.95rem;
            font-weight: 600;
            letter-spacing: 0.01em;
        }}

        .stat-card-value {{
            font-family: 'Space Grotesk', system-ui, sans-serif;
            font-size: 2.6rem;
            font-weight: 800;
            letter-spacing: -0.02em;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
