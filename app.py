import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import brentq
import numpy as np
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
.section-header {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    color: #5b8dee;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    margin: 0 0 0.5rem 0;
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
    st.error(f"Could not find `gcv_data.csv`.\n\nExpected path: `{DATA_FILE}`")
    st.stop()

# ── Column selection ──────────────────────────────────────────────────────────
df = raw[[
    "dim_agent", "dim_customer", "dim_product",
    "dim_agent_field", "dim_customer_field", "dim_product_field",
    "avg_gcv", "avg_derived_ltv", "total_orders",
    "activation_rate_pct", "first_payment_rate_pct",
    "avg_survival_sum", "combo_key",
    "gcv_pct_rank", "ltv_pct_rank",
    "avg_gap", "avg_gap_pct_of_gcv",
    "orders_gcv_above_ltv", "orders_ltv_above_gcv",
    "total_gcv", "total_derived_ltv", "total_gap",
    "avg_upfront_realized", "avg_residual_per_month",
    "avg_expected_trailing", "pct_rank_delta",
]].copy()

df.columns = [
    "agent", "customer", "product",
    "agent_field", "customer_field", "product_field",
    "avg_gcv", "avg_ltv", "total_orders",
    "activation_rate", "first_payment_rate",
    "avg_survival_sum", "combo_key",
    "gcv_pct_rank", "ltv_pct_rank",
    "avg_gap", "avg_gap_pct_of_gcv",
    "orders_gcv_above_ltv", "orders_ltv_above_gcv",
    "total_gcv", "total_ltv", "total_gap",
    "avg_upfront_realized", "avg_residual_per_month",
    "avg_expected_trailing", "pct_rank_delta",
]

numeric_cols = [
    "avg_gcv", "avg_ltv", "total_orders", "gcv_pct_rank", "ltv_pct_rank",
    "activation_rate", "first_payment_rate", "avg_survival_sum",
    "avg_gap", "avg_gap_pct_of_gcv", "orders_gcv_above_ltv", "orders_ltv_above_gcv",
    "total_gcv", "total_ltv", "total_gap",
    "avg_upfront_realized", "avg_residual_per_month", "avg_expected_trailing", "pct_rank_delta",
]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["avg_gcv", "avg_ltv", "total_orders", "gcv_pct_rank", "ltv_pct_rank",
                        "activation_rate", "first_payment_rate", "avg_survival_sum"])

# ── Survival curve helper ─────────────────────────────────────────────────────
def compute_survival_pct(activation_rate, first_payment_rate, avg_survival_sum, month):
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

def hex_to_rgba(hex_color, alpha=0.2):
    """Convert a #rrggbb hex string to rgba() that Plotly accepts."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

SURVIVAL_OPTIONS = {
    "Activation":   "activation",
    "Bill Pay":     "billpay",
    "Month 1":  1,  "Month 2":  2,  "Month 3":  3,
    "Month 4":  4,  "Month 5":  5,  "Month 6":  6,
    "Month 7":  7,  "Month 8":  8,  "Month 9":  9,
    "Month 10": 10, "Month 11": 11, "Month 12": 12,
}

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# GCV vs LTV Pipeline")
st.markdown(
    "<p style='font-family:DM Mono,monospace;font-size:0.72rem;color:#6b7280;"
    "letter-spacing:0.08em;text-transform:uppercase;margin-top:-12px;margin-bottom:8px;'>"
    "GCV Percentile Rank vs LTV Percentile Rank · Jul–Oct 2025</p>",
    unsafe_allow_html=True,
)
st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

# ── Shared filters — row 1: category selectors + color by ───────────────────
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("<p class='plot-label'>Agent Category</p>", unsafe_allow_html=True)
    sel_agent = st.selectbox("Agent Category", agent_options, label_visibility="collapsed")

with col2:
    st.markdown("<p class='plot-label'>Customer Category</p>", unsafe_allow_html=True)
    sel_customer = st.selectbox("Customer Category", customer_options, label_visibility="collapsed")

with col3:
    st.markdown("<p class='plot-label'>Product Type</p>", unsafe_allow_html=True)
    product_values = ["All"] + sorted(df["product"].dropna().unique().tolist())
    sel_product = st.selectbox("Product Type", product_values, label_visibility="collapsed")

with col4:
    st.markdown("<p class='plot-label'>Color By</p>", unsafe_allow_html=True)
    sel_color_by = st.selectbox("Color By", list(COLOR_BY_OPTIONS.keys()), label_visibility="collapsed")

# ── Shared filters — row 2: value sub-filters (context-sensitive) ────────────
# Derive available values for each dimension given the category selection above
def _vals_for(field_col, val_col, field_key):
    """Return sorted unique values in val_col where field_col == field_key."""
    return sorted(df[df[field_col] == field_key][val_col].dropna().unique().tolist())

if sel_agent != "All":
    agent_field_key  = AGENT_LABEL_TO_FIELD[sel_agent]
    agent_val_opts   = ["All"] + _vals_for("agent_field", "agent", agent_field_key)
else:
    agent_val_opts   = ["All"] + sorted(df["agent"].dropna().unique().tolist())

if sel_customer != "All":
    customer_field_key  = CUSTOMER_LABEL_TO_FIELD[sel_customer]
    customer_val_opts   = ["All"] + _vals_for("customer_field", "customer", customer_field_key)
else:
    customer_val_opts   = ["All"] + sorted(df["customer"].dropna().unique().tolist())

v1, v2, v3 = st.columns(3)

with v1:
    label = f"Agent Value{' · ' + sel_agent if sel_agent != 'All' else ''}"
    st.markdown(f"<p class='plot-label'>{label}</p>", unsafe_allow_html=True)
    sel_agent_val = st.multiselect(
        "Agent Value", agent_val_opts[1:],
        default=None,
        placeholder="All values",
        label_visibility="collapsed",
    )

with v2:
    label = f"Customer Value{' · ' + sel_customer if sel_customer != 'All' else ''}"
    st.markdown(f"<p class='plot-label'>{label}</p>", unsafe_allow_html=True)
    sel_customer_val = st.multiselect(
        "Customer Value", customer_val_opts[1:],
        default=None,
        placeholder="All values",
        label_visibility="collapsed",
    )

with v3:
    st.markdown("<p class='plot-label'>Product Value</p>", unsafe_allow_html=True)
    all_product_vals = sorted(df["product"].dropna().unique().tolist())
    # if a product type category filter is active, restrict options
    if sel_product != "All":
        all_product_vals = [sel_product]
    sel_product_val = st.multiselect(
        "Product Value", all_product_vals,
        default=None,
        placeholder="All values",
        label_visibility="collapsed",
    )

# ── Filter data ───────────────────────────────────────────────────────────────
filtered = df.copy()

# Category-level filters
if sel_agent != "All":
    filtered = filtered[filtered["agent_field"] == AGENT_LABEL_TO_FIELD[sel_agent]]
if sel_customer != "All":
    filtered = filtered[filtered["customer_field"] == CUSTOMER_LABEL_TO_FIELD[sel_customer]]
if sel_product != "All":
    filtered = filtered[filtered["product"] == sel_product]

# Value-level sub-filters
if sel_agent_val:
    filtered = filtered[filtered["agent"].isin(sel_agent_val)]
if sel_customer_val:
    filtered = filtered[filtered["customer"].isin(sel_customer_val)]
if sel_product_val:
    filtered = filtered[filtered["product"].isin(sel_product_val)]

# ── Stats row ─────────────────────────────────────────────────────────────────
st.markdown("<div style='margin: 1rem 0 0.5rem 0;'></div>", unsafe_allow_html=True)
s1, s2, s3, s4, s5 = st.columns(5)

n_points  = len(filtered)
avg_gcv   = filtered["avg_gcv"].mean()  if n_points > 0 else None
avg_ltv   = filtered["avg_ltv"].mean()  if n_points > 0 else None
avg_gap   = filtered["avg_gap"].mean()  if n_points > 0 else None
avg_act   = filtered["activation_rate"].mean() if n_points > 0 else None

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
with s4:
    gap_sign = "+" if (avg_gap or 0) > 0 else ""
    val = f"{gap_sign}${avg_gap:.0f}" if avg_gap is not None else "—"
    color = "#e05c8a" if (avg_gap or 0) > 0 else "#3dd68c"
    st.markdown(f"""<div class="stat-container">
        <div class="stat-value" style="color:{color};">{val}</div>
        <div class="stat-label">Avg GCV–LTV Gap</div>
    </div>""", unsafe_allow_html=True)
with s5:
    val = f"{avg_act:.1f}%" if avg_act is not None else "—"
    st.markdown(f"""<div class="stat-container">
        <div class="stat-value">{val}</div>
        <div class="stat-label">Avg Activation Rate</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<div style='margin-bottom:1.5rem;'></div>", unsafe_allow_html=True)

# ── Shared color map ──────────────────────────────────────────────────────────
color_col    = COLOR_BY_OPTIONS[sel_color_by]
color_values = sorted(filtered[color_col].dropna().unique().tolist())
color_map    = {val: PALETTES[i % len(PALETTES)] for i, val in enumerate(color_values)}

COMMON_LEGEND = dict(
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
)

COMMON_LAYOUT = dict(
    paper_bgcolor="#0d0f14",
    plot_bgcolor="#13161e",
    font=dict(family="DM Mono, monospace", color="#6b7280", size=11),
    margin=dict(l=60, r=30, t=30, b=60),
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
# SECTION 1 — Rank Comparison
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("<p class='section-header'>01 · Rank Comparison</p>", unsafe_allow_html=True)
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
        height=480,
        legend=COMMON_LEGEND,
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
# SECTION 2 — Survival Rate vs Avg GCV
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
st.markdown("<p class='section-header'>02 · Survival Analysis</p>", unsafe_allow_html=True)
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
    y_vals = filtered[surv_col].dropna()
    y_min  = (int(max(0, y_vals.min() - 3)) // 5) * 5
    y_max  = min(102, ((int(y_vals.max() + 3) // 5) + 1) * 5)
    x_vals = filtered["avg_gcv"].dropna()
    x_min  = max(0, x_vals.min() - 5)
    x_max  = x_vals.max() + 5

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
        height=480,
        legend=COMMON_LEGEND,
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


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Survival Curve Fan (NEW)
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
st.markdown("### Survival Curve Fan")
st.markdown(
    "<p style='font-family:DM Mono,monospace;font-size:0.68rem;color:#6b7280;margin-top:-8px;margin-bottom:16px;'>"
    "Monthly survival trajectory per group — activation through month 12</p>",
    unsafe_allow_html=True,
)

if n_points == 0:
    st.warning("No data matches the selected filters.")
else:
    # Aggregate by color dimension — weighted average of survival by total_orders
    surv_stage_cols = ["activation", "billpay"] + list(range(1, 13))
    surv_x_labels   = ["Act.", "Bill\nPay"] + [f"M{n}" for n in range(1, 13)]

    fig3 = go.Figure()

    # Add a shaded reference band for overall median trajectory
    all_surv_vals = []
    for stage in surv_stage_cols:
        col = f"surv_{stage}"
        all_surv_vals.append(filtered[col].median())

    fig3.add_trace(go.Scatter(
        x=surv_x_labels,
        y=all_surv_vals,
        mode="lines",
        name="Overall Median",
        line=dict(color="#3a4060", width=2, dash="dot"),
        hovertemplate="Overall median: %{y:.1f}%<extra></extra>",
    ))

    for val in color_values:
        subset = filtered[filtered[color_col] == val]
        if subset.empty:
            continue
        # Weighted mean by total_orders
        weights = subset["total_orders"]
        curve   = []
        for stage in surv_stage_cols:
            col   = f"surv_{stage}"
            wvals = subset[col]
            wmean = (wvals * weights).sum() / weights.sum() if weights.sum() > 0 else wvals.mean()
            curve.append(wmean)

        fig3.add_trace(go.Scatter(
            x=surv_x_labels,
            y=curve,
            mode="lines+markers",
            name=str(val),
            line=dict(color=color_map[val], width=2.5),
            marker=dict(color=color_map[val], size=7),
            hovertemplate=f"<b>{val}</b><br>Stage: %{{x}}<br>Survival: %{{y:.1f}}%<extra></extra>",
        ))

    fig3.update_layout(
        **COMMON_LAYOUT,
        height=420,
        legend=COMMON_LEGEND,
        xaxis=dict(
            title=dict(text="SURVIVAL STAGE", font=dict(size=11, color="#6b7280")),
            **AXIS_STYLE,
        ),
        yaxis=dict(
            title=dict(text="SURVIVAL RATE (%)", font=dict(size=11, color="#6b7280")),
            range=[0, 105],
            ticksuffix="%",
            **AXIS_STYLE,
        ),
    )
    st.plotly_chart(fig3, use_container_width=True)


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    "<p style='font-family:DM Mono,monospace;font-size:0.65rem;color:#3a4060;"
    "text-transform:uppercase;letter-spacing:0.08em;text-align:center;margin-top:2rem;'>"
    f"Loaded {len(df):,} rows from gcv_data.csv</p>",
    unsafe_allow_html=True,
)