import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from scipy.optimize import brentq
import os

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GCV vs LTV Pipeline Dashboard",
    page_icon="📊",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@400;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Syne', sans-serif; }
.main { background-color: #0d0f14; }
.block-container { padding-top: 2rem; padding-bottom: 2rem; }
h1 {
    font-family: 'Syne', sans-serif !important;
    font-weight: 800 !important;
    letter-spacing: -0.03em !important;
    color: #e8eaf0 !important;
}
h3 {
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    color: #e8eaf0 !important;
    margin-bottom: 0.25rem !important;
}
.stat-container {
    background: #13161e;
    border: 1px solid #252a3a;
    border-radius: 10px;
    padding: 16px 20px;
    text-align: center;
}
.stat-value {
    font-family: 'Syne', sans-serif;
    font-size: 1.8rem;
    font-weight: 700;
    color: #e8eaf0;
    letter-spacing: -0.02em;
    line-height: 1.1;
}
.stat-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 4px;
}
.section-divider {
    border: none;
    border-top: 1px solid #252a3a;
    margin: 0.75rem 0 1.5rem 0;
}
.plot-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 4px;
}
</style>
""", unsafe_allow_html=True)

# ── Load data ─────────────────────────────────────────────────────────────────
DATA_FILE = os.path.join(os.path.dirname(__file__), "gcv_data.csv")

@st.cache_data
def load_data(path):
    return pd.read_csv(path)

try:
    raw = load_data(DATA_FILE)
except FileNotFoundError:
    st.error(f"Could not find `gcv_data.csv` in the same folder as this script.\n\nExpected path: `{DATA_FILE}`")
    st.stop()

# ── Column selection ──────────────────────────────────────────────────────────
df = raw[[
    "dim_agent", "dim_customer", "dim_product",
    "dim_agent_field", "dim_customer_field", "dim_product_field",
    "avg_gcv", "avg_derived_ltv", "total_orders",
    "activation_rate_pct", "first_payment_rate_pct",
    "avg_survival_sum", "combo_key",
    "gcv_pct_rank", "ltv_pct_rank",
]].copy()

df.columns = [
    "agent", "customer", "product",
    "agent_field", "customer_field", "product_field",
    "avg_gcv", "avg_ltv", "total_orders",
    "activation_rate", "first_payment_rate",
    "avg_survival_sum", "combo_key",
    "gcv_pct_rank", "ltv_pct_rank",
]

for col in ["avg_gcv", "avg_ltv", "total_orders", "gcv_pct_rank", "ltv_pct_rank",
            "activation_rate", "first_payment_rate", "avg_survival_sum"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["avg_gcv", "avg_ltv", "total_orders", "gcv_pct_rank", "ltv_pct_rank",
                        "activation_rate", "first_payment_rate", "avg_survival_sum"])

# ── Survival curve helper ─────────────────────────────────────────────────────
def compute_survival_pct(activation_rate, first_payment_rate, avg_survival_sum, month):
    """
    Returns survival % for the given month label.
    'activation' -> activation_rate_pct
    'billpay'    -> first_payment_rate_pct (month 1 equivalent)
    1..12        -> geometric decay calibrated so that
                   sum(S(1)..S(12)) == avg_survival_sum
    """
    if month == "activation":
        return activation_rate
    if month == "billpay":
        return first_payment_rate

    fpr = first_payment_rate / 100.0
    s_sum = avg_survival_sum

    def f(p):
        if abs(p - 1.0) < 1e-9:
            return fpr * 12.0 - s_sum
        return fpr * (1.0 - p ** 12) / (1.0 - p) - s_sum

    try:
        p = min(brentq(f, 0.5, 1.05), 1.0)
    except Exception:
        p = 1.0

    return fpr * (p ** (month - 1)) * 100.0


@st.cache_data
def add_survival_columns(df_in):
    """Pre-compute all survival columns once."""
    df_out = df_in.copy()
    for month in ["activation", "billpay"] + list(range(1, 13)):
        col = f"surv_{month}"
        df_out[col] = df_out.apply(
            lambda r: compute_survival_pct(
                r["activation_rate"], r["first_payment_rate"], r["avg_survival_sum"], month
            ),
            axis=1,
        )
    return df_out

df = add_survival_columns(df)

# ── Category field label maps ─────────────────────────────────────────────────
AGENT_FIELD_LABELS = {
    "center_location":      "Center Location",
    "performance_quartile": "Performance Quartile",
    "tenure_category":      "Tenure Category",
}
CUSTOMER_FIELD_LABELS = {
    "brand_category":     "Brand / Non-Brand",
    "mover_switcher":     "Mover / Switcher",
    "site_serp_category": "Site / SERP",
}

AGENT_LABEL_TO_FIELD    = {v: k for k, v in AGENT_FIELD_LABELS.items()}
CUSTOMER_LABEL_TO_FIELD = {v: k for k, v in CUSTOMER_FIELD_LABELS.items()}

agent_options    = ["All"] + [AGENT_FIELD_LABELS[f]    for f in sorted(df["agent_field"].dropna().unique())    if f in AGENT_FIELD_LABELS]
customer_options = ["All"] + [CUSTOMER_FIELD_LABELS[f] for f in sorted(df["customer_field"].dropna().unique()) if f in CUSTOMER_FIELD_LABELS]

COLOR_BY_OPTIONS = {
    "Agent":        "agent",
    "Customer":     "customer",
    "Product Type": "product",
}

PALETTES = [
    "#5b8dee", "#f59e42", "#3dd68c", "#e05c8a", "#a78bfa",
    "#f87171", "#34d399", "#fbbf24", "#60a5fa", "#c084fc",
]

SURVIVAL_OPTIONS = {
    "Activation":   "activation",
    "Bill Pay":     "billpay",
    "Month 1":  1,  "Month 2":  2,  "Month 3":  3,
    "Month 4":  4,  "Month 5":  5,  "Month 6":  6,
    "Month 7":  7,  "Month 8":  8,  "Month 9":  9,
    "Month 10": 10, "Month 11": 11, "Month 12": 12,
}

# Y-axis range by survival stage (data-driven, with a little padding)
YAXIS_RANGES = {
    "activation": [75, 102],
    "billpay":    [50, 102],
    **{n: [max(0, round((35 - n * 1.5) / 5) * 5), 102] for n in range(1, 13)},
}
# Override with tighter data-driven bounds
YAXIS_RANGES.update({
    1:  [50, 102],
    2:  [45, 100],
    3:  [43, 98],
    4:  [42, 96],
    5:  [40, 95],
    6:  [38, 93],
    7:  [35, 91],
    8:  [33, 90],
    9:  [30, 88],
    10: [25, 87],
    11: [20, 86],
    12: [12, 88],
})

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# GCV vs LTV Pipeline")
st.markdown(
    "<p style='font-family:DM Mono,monospace;font-size:0.72rem;color:#6b7280;"
    "letter-spacing:0.08em;text-transform:uppercase;margin-top:-12px;margin-bottom:8px;'>"
    "GCV Percentile Rank vs LTV Percentile Rank · Jul–Oct 2025</p>",
    unsafe_allow_html=True,
)
st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

# ── Shared filters row ────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("<p class='plot-label'>Agent Category</p>", unsafe_allow_html=True)
    sel_agent = st.selectbox("Agent Category", agent_options, label_visibility="collapsed")

with col2:
    st.markdown("<p class='plot-label'>Customer Category</p>", unsafe_allow_html=True)
    sel_customer = st.selectbox("Customer Category", customer_options, label_visibility="collapsed")

with col3:
    st.markdown("<p class='plot-label'>Color By</p>", unsafe_allow_html=True)
    sel_color_by = st.selectbox("Color By", list(COLOR_BY_OPTIONS.keys()), label_visibility="collapsed")

# ── Filter data ───────────────────────────────────────────────────────────────
filtered = df.copy()
if sel_agent != "All":
    filtered = filtered[filtered["agent_field"] == AGENT_LABEL_TO_FIELD[sel_agent]]
if sel_customer != "All":
    filtered = filtered[filtered["customer_field"] == CUSTOMER_LABEL_TO_FIELD[sel_customer]]

# ── Stats row ─────────────────────────────────────────────────────────────────
st.markdown("<div style='margin: 1rem 0 0.5rem 0;'></div>", unsafe_allow_html=True)
s1, s2, s3 = st.columns(3)

n_points = len(filtered)
avg_gcv  = filtered["avg_gcv"].mean() if n_points > 0 else None
avg_ltv  = filtered["avg_ltv"].mean() if n_points > 0 else None

with s1:
    st.markdown(f"""<div class="stat-container">
        <div class="stat-value">{n_points}</div>
        <div class="stat-label">Data Points</div>
    </div>""", unsafe_allow_html=True)
with s2:
    val = f"${avg_gcv:.0f}" if avg_gcv is not None else "—"
    st.markdown(f"""<div class="stat-container">
        <div class="stat-value">{val}</div>
        <div class="stat-label">Avg GCV</div>
    </div>""", unsafe_allow_html=True)
with s3:
    val = f"${avg_ltv:.0f}" if avg_ltv is not None else "—"
    st.markdown(f"""<div class="stat-container">
        <div class="stat-value">{val}</div>
        <div class="stat-label">Avg LTV</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<div style='margin-bottom:1.5rem;'></div>", unsafe_allow_html=True)

# ── Shared color map ──────────────────────────────────────────────────────────
color_col    = COLOR_BY_OPTIONS[sel_color_by]
color_values = sorted(filtered[color_col].dropna().unique().tolist())
color_map    = {val: PALETTES[i % len(PALETTES)] for i, val in enumerate(color_values)}

COMMON_LAYOUT = dict(
    paper_bgcolor="#0d0f14",
    plot_bgcolor="#13161e",
    font=dict(family="DM Mono, monospace", color="#6b7280", size=11),
    legend=dict(
        title=dict(
            text=sel_color_by.upper(),
            font=dict(size=10, color="#6b7280", family="DM Mono, monospace"),
        ),
        bgcolor="#13161e",
        bordercolor="#252a3a",
        borderwidth=1,
        font=dict(color="#e8eaf0", family="Syne, sans-serif", size=12),
        itemclick="toggle",
        itemdoubleclick="toggleothers",
    ),
    margin=dict(l=60, r=30, t=30, b=60),
    height=480,
    hoverlabel=dict(
        bgcolor="#1a1f2e",
        bordercolor="#252a3a",
        font=dict(family="DM Mono, monospace", color="#e8eaf0", size=12),
    ),
)

AXIS_STYLE = dict(
    gridcolor="#1c2030",
    linecolor="#252a3a",
    tickcolor="#252a3a",
    tickfont=dict(color="#6b7280"),
)

# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 1 — GCV Percentile vs LTV Percentile
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("### GCV Percentile vs LTV Percentile")

if n_points == 0:
    st.warning("No data matches the selected filters.")
else:
    fig1 = go.Figure()

    fig1.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        line=dict(color="#3a4060", width=1.5, dash="dash"),
        hoverinfo="skip", showlegend=False,
    ))

    for val in color_values:
        subset = filtered[filtered[color_col] == val]
        if subset.empty:
            continue
        hover_text = [
            f"<b>{r.agent} · {r.customer} · {r.product}</b><br>"
            f"GCV Normalized: {r.gcv_pct_rank:.4f}<br>"
            f"LTV Normalized: {r.ltv_pct_rank:.4f}<br>"
            f"Avg GCV: ${r.avg_gcv:.2f}<br>"
            f"Avg LTV: ${r.avg_ltv:.2f}<br>"
            f"Total Orders: {int(r.total_orders):,}<br>"
            f"Activation Rate: {r.activation_rate:.1f}%<br>"
            f"Grouping: {r.combo_key}"
            for r in subset.itertuples()
        ]
        fig1.add_trace(go.Scatter(
            x=subset["gcv_pct_rank"],
            y=subset["ltv_pct_rank"],
            mode="markers",
            name=str(val),
            marker=dict(color=color_map[val], size=11, opacity=0.92, line=dict(width=0)),
            text=hover_text,
            hovertemplate="%{text}<extra></extra>",
        ))

    fig1.update_layout(
        **COMMON_LAYOUT,
        xaxis=dict(title=dict(text="GCV NORMALIZED RANK (0–1)", font=dict(size=11, color="#6b7280")),
                   range=[-0.02, 1.02], **AXIS_STYLE),
        yaxis=dict(title=dict(text="LTV NORMALIZED RANK (0–1)", font=dict(size=11, color="#6b7280")),
                   range=[-0.02, 1.02], **AXIS_STYLE),
        annotations=[dict(
            x=0.98, y=1.01, text="GCV = LTV",
            showarrow=False,
            font=dict(size=10, color="#3a4060", family="DM Mono, monospace"),
            xanchor="right",
        )],
    )
    st.plotly_chart(fig1, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 2 — Survival Rate vs Avg GCV
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
st.markdown("### Survival Rate vs Avg GCV")

surv_col2, _ = st.columns([1, 2])
with surv_col2:
    st.markdown("<p class='plot-label'>Survival Stage</p>", unsafe_allow_html=True)
    sel_survival = st.selectbox(
        "Survival Stage",
        list(SURVIVAL_OPTIONS.keys()),
        index=0,
        label_visibility="collapsed",
    )

surv_month = SURVIVAL_OPTIONS[sel_survival]
surv_col   = f"surv_{surv_month}"

if n_points == 0:
    st.warning("No data matches the selected filters.")
else:
    # Compute y-axis range from actual filtered data with padding
    y_vals   = filtered[surv_col].dropna()
    y_min    = max(0, y_vals.min() - 3)
    y_max    = min(102, y_vals.max() + 3)
    # Round to nearest 5 for clean ticks
    y_min    = (int(y_min) // 5) * 5
    y_max    = min(102, ((int(y_max) // 5) + 1) * 5)

    x_vals   = filtered["avg_gcv"].dropna()
    x_min    = max(0, x_vals.min() - 5)
    x_max    = x_vals.max() + 5

    fig2 = go.Figure()

    for val in color_values:
        subset = filtered[filtered[color_col] == val]
        if subset.empty:
            continue
        hover_text = [
            f"<b>{r.agent} · {r.customer} · {r.product}</b><br>"
            f"Avg GCV: ${r.avg_gcv:.2f}<br>"
            f"{sel_survival} Survival: {getattr(r, surv_col):.1f}%<br>"
            f"Total Orders: {int(r.total_orders):,}<br>"
            f"Grouping: {r.combo_key}"
            for r in subset.itertuples()
        ]
        fig2.add_trace(go.Scatter(
            x=subset["avg_gcv"],
            y=subset[surv_col],
            mode="markers",
            name=str(val),
            marker=dict(color=color_map[val], size=11, opacity=0.92, line=dict(width=0)),
            text=hover_text,
            hovertemplate="%{text}<extra></extra>",
            showlegend=True,
        ))

    fig2.update_layout(
        **COMMON_LAYOUT,
        xaxis=dict(
            title=dict(text="AVG GCV ($)", font=dict(size=11, color="#6b7280")),
            range=[x_min, x_max],
            tickprefix="$",
            **AXIS_STYLE,
        ),
        yaxis=dict(
            title=dict(text=f"{sel_survival.upper()} SURVIVAL RATE (%)", font=dict(size=11, color="#6b7280")),
            range=[y_min, y_max],
            ticksuffix="%",
            **AXIS_STYLE,
        ),
    )
    st.plotly_chart(fig2, use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    "<p style='font-family:DM Mono,monospace;font-size:0.65rem;color:#3a4060;"
    "text-transform:uppercase;letter-spacing:0.08em;text-align:center;margin-top:2rem;'>"
    f"Loaded {len(df):,} rows from gcv_data.csv</p>",
    unsafe_allow_html=True,
)