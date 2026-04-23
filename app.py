import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import os
from PIL import Image

# ── Page config ───────────────────────────────────────────────────────────────
icon = Image.open(os.path.join(os.path.dirname(__file__), "logo.png")).convert("RGBA")
data = icon.getdata()
new_data = [
    (r, g, b, 0) if (r > 200 and g > 200 and b > 200) else (r, g, b, a)
    for r, g, b, a in data
]
icon.putdata(new_data)

st.set_page_config(
    page_title="GCV vs LTV Pipeline Dashboard",
    page_icon=icon,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@400;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Syne', sans-serif; }
.main { background-color: #0d0f14; }
.block-container { padding-top: 2rem; padding-bottom: 2rem; }
[data-testid="stSidebar"] {
    background-color: #0d0f14;
    border-right: 1px solid #252a3a;
}
[data-testid="stSidebar"] .block-container { padding-top: 1.5rem; }
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
.sidebar-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 3px;
    margin-top: 12px;
}
.sidebar-section {
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    color: #5b8dee;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    margin: 1.25rem 0 0.4rem 0;
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

# =============================================================================
# LOAD DATA
# =============================================================================

LTV_FILE      = os.path.join(os.path.dirname(__file__), "ltv_call_level.csv")
SURVIVAL_FILE = os.path.join(os.path.dirname(__file__), "survival_call_level.csv")

LTV_SURV_COLS  = [f"survived_m{m}" for m in range(1, 7)]
SURV_COLS      = [f"survived_m{m}" for m in range(1, 13)]

@st.cache_data(ttl=3600)
def load_ltv(path):
    df = pd.read_csv(path)
    df = df[df["partner_name"] != "Atlantex Power"]          # exclude Atlantex
    for col in ["order_date_est", "activation_date", "first_payment_date", "end_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    num_cols = [
        "gcv", "gcv_v2", "derived_ltv", "trailing_revenue", "gcv_ltv_gap",
        "ltv_upfront_realized", "ltv_upfront_bounty", "mil_rate",
        "avg_monthly_usage", "observed_tenure_days", "months_on_plan",
        "activated_ind", "first_payment_ind", "active_ind", "churned_ind",
    ] + LTV_SURV_COLS + [f"residual_m{m}" for m in range(1, 7)]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

@st.cache_data(ttl=3600)
def load_survival(path):
    df = pd.read_csv(path)
    df = df[df["partner_name"] != "Atlantex Power"]          # exclude Atlantex
    for col in ["order_date_est", "call_date", "activation_date",
                "first_payment_date", "end_date", "hire_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    num_cols = [
        "gcv_v2", "avg_monthly_usage", "observed_tenure_days",
        "activated_ind", "first_payment_ind", "m6_to_m12_completion",
    ] + SURV_COLS
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

for fpath, fname in [(LTV_FILE, "ltv_call_level.csv"), (SURVIVAL_FILE, "survival_call_level.csv")]:
    if not os.path.exists(fpath):
        st.error(f"Could not find `{fname}`.\n\nExpected path: `{fpath}`")
        st.stop()

ltv_raw      = load_ltv(LTV_FILE)
survival_raw = load_survival(SURVIVAL_FILE)

# =============================================================================
# DIMENSION MAPS
# =============================================================================

AGENT_DIMS = {
    "Center Location":     "center_location",
    "Conversion Quartile": "conversion_quartile",
    "Survival Quartile":   "survival_quartile",
    "Tenure":              "tenure_category",
}
PRODUCT_DIMS = {
    "Plan Type": "sold_product_type",
    "Provider":  "partner_name",
}
CUSTOMER_DIMS = {
    "Mover / Switcher": "mover_switcher",
    "Brand Category":   "brand_category",
    "Site / SERP":      "site_serp_category",
    "Usage Band":       "usage_band",
    "Consistent Usage": "consistent_usage",
}

COLOR_BY_OPTIONS = {
    "Agent":    "agent_dim",
    "Product":  "product_dim",
    "Customer": "customer_dim",
}

PALETTES = [
    "#5b8dee", "#f59e42", "#3dd68c", "#e05c8a", "#a78bfa",
    "#f87171", "#34d399", "#fbbf24", "#60a5fa", "#c084fc",
]

SURVIVAL_OPTIONS = {
    "Month 1":  1,  "Month 2":  2,  "Month 3":  3,
    "Month 4":  4,  "Month 5":  5,  "Month 6":  6,
    "Month 7":  7,  "Month 8":  8,  "Month 9":  9,
    "Month 10": 10, "Month 11": 11, "Month 12": 12,
}

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

def make_legend(title_text):
    return dict(
        title=dict(
            text=title_text.upper(),
            font=dict(size=10, color="#6b7280", family="DM Mono, monospace"),
        ),
        bgcolor="#13161e",
        bordercolor="#252a3a",
        borderwidth=1,
        font=dict(color="#e8eaf0", family="Syne, sans-serif", size=12),
        itemclick="toggle",
        itemdoubleclick="toggleothers",
    )

# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.markdown("# Filters")
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    st.markdown("<p class='sidebar-section'>Grouping</p>", unsafe_allow_html=True)

    st.markdown("<p class='sidebar-label'>Agent Dimension</p>", unsafe_allow_html=True)
    sel_agent_dim = st.selectbox(
        "Agent Dimension", list(AGENT_DIMS.keys()), index=0,
        label_visibility="collapsed",
    )

    st.markdown("<p class='sidebar-label'>Product Dimension</p>", unsafe_allow_html=True)
    sel_product_dim = st.selectbox(
        "Product Dimension", list(PRODUCT_DIMS.keys()), index=0,
        label_visibility="collapsed",
    )

    st.markdown("<p class='sidebar-label'>Customer Dimension</p>", unsafe_allow_html=True)
    sel_customer_dim = st.selectbox(
        "Customer Dimension", list(CUSTOMER_DIMS.keys()), index=0,
        label_visibility="collapsed",
    )

    st.markdown("<p class='sidebar-label'>Color By</p>", unsafe_allow_html=True)
    sel_color_by = st.selectbox(
        "Color By", list(COLOR_BY_OPTIONS.keys()), index=2,
        label_visibility="collapsed",
    )

    agent_col    = AGENT_DIMS[sel_agent_dim]
    product_col  = PRODUCT_DIMS[sel_product_dim]
    customer_col = CUSTOMER_DIMS[sel_customer_dim]

    st.markdown("<p class='sidebar-section'>Filter Values</p>", unsafe_allow_html=True)

    def _opts(df, col):
        return sorted(df[col].dropna().unique().tolist()) if col in df.columns else []

    st.markdown("<p class='sidebar-label'>Center</p>", unsafe_allow_html=True)
    sel_center = st.multiselect("Center", _opts(ltv_raw, "center_location"),
                                default=[], placeholder="All", label_visibility="collapsed")

    st.markdown("<p class='sidebar-label'>Provider</p>", unsafe_allow_html=True)
    sel_provider = st.multiselect("Provider", _opts(ltv_raw, "partner_name"),
                                  default=[], placeholder="All", label_visibility="collapsed")

    st.markdown("<p class='sidebar-label'>Plan Type</p>", unsafe_allow_html=True)
    sel_plan = st.multiselect("Plan Type", _opts(ltv_raw, "sold_product_type"),
                              default=[], placeholder="All", label_visibility="collapsed")

    st.markdown("<p class='sidebar-label'>Mover / Switcher</p>", unsafe_allow_html=True)
    sel_mov = st.multiselect("Mover / Switcher", _opts(ltv_raw, "mover_switcher"),
                             default=[], placeholder="All", label_visibility="collapsed")

    st.markdown("<p class='sidebar-label'>Brand Category</p>", unsafe_allow_html=True)
    sel_brand = st.multiselect("Brand Category", _opts(ltv_raw, "brand_category"),
                               default=[], placeholder="All", label_visibility="collapsed")

    st.markdown("<p class='sidebar-label'>Site / SERP</p>", unsafe_allow_html=True)
    sel_serp = st.multiselect("Site / SERP", _opts(ltv_raw, "site_serp_category"),
                              default=[], placeholder="All", label_visibility="collapsed")

    st.markdown("<p class='sidebar-label'>Conversion Quartile</p>", unsafe_allow_html=True)
    sel_cq = st.multiselect("Conversion Quartile", _opts(ltv_raw, "conversion_quartile"),
                            default=[], placeholder="All", label_visibility="collapsed")

    st.markdown("<p class='sidebar-label'>Survival Quartile</p>", unsafe_allow_html=True)
    sel_sq = st.multiselect("Survival Quartile", _opts(ltv_raw, "survival_quartile"),
                            default=[], placeholder="All", label_visibility="collapsed")

    st.markdown("<p class='sidebar-label'>Tenure</p>", unsafe_allow_html=True)
    sel_tenure = st.multiselect("Tenure", _opts(ltv_raw, "tenure_category"),
                                default=[], placeholder="All", label_visibility="collapsed")

    st.markdown("<p class='sidebar-label'>Usage Band</p>", unsafe_allow_html=True)
    sel_usage = st.multiselect("Usage Band", _opts(ltv_raw, "usage_band"),
                               default=[], placeholder="All", label_visibility="collapsed")

    st.markdown("<p class='sidebar-label'>Consistent Usage</p>", unsafe_allow_html=True)
    sel_consist = st.multiselect("Consistent Usage", _opts(ltv_raw, "consistent_usage"),
                                 default=[], placeholder="All", label_visibility="collapsed")

    st.markdown("<div style='margin-top:2rem;'></div>", unsafe_allow_html=True)
    st.markdown(
        f"<p style='font-family:DM Mono,monospace;font-size:0.6rem;color:#3a4060;"
        f"text-transform:uppercase;letter-spacing:0.08em;'>"
        f"{len(ltv_raw):,} LTV rows · {len(survival_raw):,} survival rows</p>",
        unsafe_allow_html=True,
    )

# =============================================================================
# APPLY FILTERS
# =============================================================================

FILTER_MAP = [
    ("center_location",    sel_center),
    ("partner_name",       sel_provider),
    ("sold_product_type",  sel_plan),
    ("mover_switcher",     sel_mov),
    ("brand_category",     sel_brand),
    ("site_serp_category", sel_serp),
    ("conversion_quartile", sel_cq),
    ("survival_quartile",  sel_sq),
    ("tenure_category",    sel_tenure),
    ("usage_band",         sel_usage),
    ("consistent_usage",   sel_consist),
]

def apply_filters(df):
    d = df.copy()
    for col, sel in FILTER_MAP:
        if sel and col in d.columns:
            d = d[d[col].isin(sel)]
    return d

ltv      = apply_filters(ltv_raw)
survival = apply_filters(survival_raw)

# =============================================================================
# AGGREGATE  — one row per (agent_dim × product_dim × customer_dim)
# for the scatter / survival fan which expect segment-level points
# =============================================================================

def aggregate_ltv(df, agent_col, product_col, customer_col):
    surv_means = {f"surv_m{m}": (f"survived_m{m}", "mean")
                  for m in range(1, 7) if f"survived_m{m}" in df.columns}
    agg_dict = {
        "avg_gcv":          ("gcv",                 "mean"),
        "avg_ltv":          ("derived_ltv",          "mean"),
        "avg_trailing":     ("trailing_revenue",     "mean"),
        "avg_upfront":      ("ltv_upfront_realized", "mean"),
        "avg_gap":          ("gcv_ltv_gap",          "mean"),
        "total_orders":     ("gcv",                  "count"),
        "total_gcv":        ("gcv",                  "sum"),
        "total_ltv":        ("derived_ltv",          "sum"),
        "activation_rate":  ("activated_ind",        "mean"),
        "first_pmt_rate":   ("first_payment_ind",    "mean"),
        **surv_means,
    }
    grp = (
        df.dropna(subset=[agent_col, product_col, customer_col])
        .groupby([agent_col, product_col, customer_col])
        .agg(**agg_dict)
        .reset_index()
        .rename(columns={agent_col: "agent_dim", product_col: "product_dim",
                         customer_col: "customer_dim"})
    )
    # Percentile ranks within this aggregated view
    grp["gcv_pct_rank"] = grp["avg_gcv"].rank(pct=True).round(4)
    grp["ltv_pct_rank"] = grp["avg_ltv"].rank(pct=True).round(4)
    grp["pct_rank_delta"] = (grp["gcv_pct_rank"] - grp["ltv_pct_rank"]).round(4)
    return grp

def aggregate_survival(df, agent_col, product_col, customer_col):
    surv_means = {f"surv_m{m}": (f"survived_m{m}", "mean")
                  for m in range(1, 13) if f"survived_m{m}" in df.columns}
    comp_dict  = {"m6_to_m12": ("m6_to_m12_completion", "mean")} \
                 if "m6_to_m12_completion" in df.columns else {}
    agg_dict = {
        "total_orders":    ("gcv_v2",        "count"),
        "activation_rate": ("activated_ind", "mean"),
        **surv_means,
        **comp_dict,
    }
    return (
        df.dropna(subset=[agent_col, product_col, customer_col])
        .groupby([agent_col, product_col, customer_col])
        .agg(**agg_dict)
        .reset_index()
        .rename(columns={agent_col: "agent_dim", product_col: "product_dim",
                         customer_col: "customer_dim"})
    )

ltv_agg  = aggregate_ltv(ltv,      agent_col, product_col, customer_col)
surv_agg = aggregate_survival(survival, agent_col, product_col, customer_col)

# Color map — driven by the selected color-by column
color_col    = COLOR_BY_OPTIONS[sel_color_by]
color_values = sorted(ltv_agg[color_col].dropna().unique().tolist()) \
               if color_col in ltv_agg.columns else []
color_map    = {v: PALETTES[i % len(PALETTES)] for i, v in enumerate(color_values)}

surv_color_values = sorted(surv_agg[color_col].dropna().unique().tolist()) \
                    if color_col in surv_agg.columns else []
surv_color_map    = {v: PALETTES[i % len(PALETTES)] for i, v in enumerate(surv_color_values)}

# =============================================================================
# HEADER
# =============================================================================

st.markdown("# GCV vs LTV Pipeline")
st.markdown(
    f"<p style='font-family:DM Mono,monospace;font-size:0.72rem;color:#6b7280;"
    f"letter-spacing:0.08em;text-transform:uppercase;margin-top:-12px;margin-bottom:8px;'>"
    f"{sel_agent_dim} × {sel_product_dim} × {sel_customer_dim} · "
    f"{len(ltv):,} LTV orders · {len(survival):,} survival orders</p>",
    unsafe_allow_html=True,
)
st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

# =============================================================================
# KPI STATS ROW
# =============================================================================

n_seg      = len(ltv_agg)
avg_gcv    = ltv_agg["avg_gcv"].mean()   if n_seg > 0 else None
avg_ltv    = ltv_agg["avg_ltv"].mean()   if n_seg > 0 else None
avg_gap    = ltv_agg["avg_gap"].mean()   if n_seg > 0 else None
avg_act    = ltv["activated_ind"].mean() * 100 if len(ltv) > 0 else None

s1, s2, s3, s4, s5 = st.columns(5)

with s1:
    st.markdown(f"""<div class="stat-container">
        <div class="stat-value">{n_seg:,}</div>
        <div class="stat-label">Segments</div>
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
        <div class="stat-label">Avg Derived LTV</div>
    </div>""", unsafe_allow_html=True)
with s4:
    gap_sign = "+" if (avg_gap or 0) > 0 else ""
    val      = f"{gap_sign}${avg_gap:.0f}" if avg_gap is not None else "—"
    color    = "#e05c8a" if (avg_gap or 0) > 0 else "#3dd68c"
    st.markdown(f"""<div class="stat-container">
        <div class="stat-value" style="color:{color};">{val}</div>
        <div class="stat-label">Avg GCV–LTV Gap</div>
    </div>""", unsafe_allow_html=True)
with s5:
    val = f"{avg_act:.1f}%" if avg_act is not None else "—"
    st.markdown(f"""<div class="stat-container">
        <div class="stat-value">{val}</div>
        <div class="stat-label">Activation Rate</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<div style='margin-bottom:1.5rem;'></div>", unsafe_allow_html=True)

# =============================================================================
# SECTION 1 — GCV vs LTV Rank Scatter  (LTV data)
# =============================================================================

st.markdown("<p class='section-header'>01 · Rank Comparison</p>", unsafe_allow_html=True)
st.markdown("### GCV Percentile vs LTV Percentile")

if n_seg == 0:
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
        subset = ltv_agg[ltv_agg[color_col] == val]
        if subset.empty:
            continue
        hover_text = [
            f"<b>{r.agent_dim} · {r.product_dim} · {r.customer_dim}</b><br>"
            f"GCV Rank: {r.gcv_pct_rank:.4f}<br>"
            f"LTV Rank: {r.ltv_pct_rank:.4f}<br>"
            f"Avg GCV: ${r.avg_gcv:.2f}<br>"
            f"Avg LTV: ${r.avg_ltv:.2f}<br>"
            f"Avg Gap: ${r.avg_gap:.2f}<br>"
            f"Orders: {int(r.total_orders):,}<br>"
            f"Activation: {r.activation_rate * 100:.1f}%"
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
        legend=make_legend(sel_color_by),
        xaxis=dict(
            title=dict(text="GCV NORMALIZED RANK (0–1)", font=dict(size=11, color="#6b7280")),
            range=[-0.02, 1.02], **AXIS_STYLE,
        ),
        yaxis=dict(
            title=dict(text="LTV NORMALIZED RANK (0–1)", font=dict(size=11, color="#6b7280")),
            range=[-0.02, 1.02], **AXIS_STYLE,
        ),
        annotations=[dict(
            x=0.98, y=1.01, text="GCV = LTV",
            showarrow=False,
            font=dict(size=10, color="#3a4060", family="DM Mono, monospace"),
            xanchor="right",
        )],
    )
    st.plotly_chart(fig1, use_container_width=True)


# =============================================================================
# SECTION 2 — Survival Curve Fan  (Survival data — all 12 months)
# =============================================================================

st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
st.markdown("<p class='section-header'>02 · Survival Curve Fan</p>", unsafe_allow_html=True)
st.markdown("### Survival Curve Fan")
st.markdown(
    "<p style='font-family:DM Mono,monospace;font-size:0.68rem;color:#6b7280;"
    "margin-top:-8px;margin-bottom:16px;'>"
    "Monthly survival trajectory per segment — months 1 through 12</p>",
    unsafe_allow_html=True,
)

surv_month_cols = [f"surv_m{m}" for m in range(1, 13) if f"surv_m{m}" in surv_agg.columns]

if surv_agg.empty or not surv_month_cols:
    st.warning("No survival data matches the selected filters.")
else:
    x_labels = [f"M{m}" for m in range(1, len(surv_month_cols) + 1)]

    fig2 = go.Figure()

    overall_median = [surv_agg[c].median() * 100 for c in surv_month_cols]
    fig2.add_trace(go.Scatter(
        x=x_labels, y=overall_median,
        mode="lines",
        name="Overall Median",
        line=dict(color="#3a4060", width=2, dash="dot"),
        hovertemplate="Overall median: %{y:.1f}%<extra></extra>",
    ))

    for val in surv_color_values:
        subset = surv_agg[surv_agg[color_col] == val]
        if subset.empty:
            continue
        weights = subset["total_orders"]
        w_sum   = weights.sum()
        curve = []
        for c in surv_month_cols:
            wvals = subset[c]
            wmean = (wvals * weights).sum() / w_sum if w_sum > 0 else wvals.mean()
            curve.append(wmean * 100)

        fig2.add_trace(go.Scatter(
            x=x_labels, y=curve,
            mode="lines+markers",
            name=str(val),
            line=dict(color=surv_color_map[val], width=2.5),
            marker=dict(color=surv_color_map[val], size=7),
            hovertemplate=f"<b>{val}</b><br>Month: %{{x}}<br>Survival: %{{y:.1f}}%<extra></extra>",
        ))

    fig2.update_layout(
        **COMMON_LAYOUT,
        height=420,
        legend=make_legend(sel_color_by),
        xaxis=dict(
            title=dict(text="MONTH", font=dict(size=11, color="#6b7280")),
            **AXIS_STYLE,
        ),
        yaxis=dict(
            title=dict(text="SURVIVAL RATE (%)", font=dict(size=11, color="#6b7280")),
            range=[0, 105],
            ticksuffix="%",
            **AXIS_STYLE,
        ),
    )
    st.plotly_chart(fig2, use_container_width=True)


# =============================================================================
# SECTION 3 — Survival Rate vs Avg GCV  (Survival data)
# =============================================================================

st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
st.markdown("<p class='section-header'>03 · Survival Analysis</p>", unsafe_allow_html=True)
st.markdown("### Survival Rate vs Avg GCV")

surv_col3, _ = st.columns([1, 2])
with surv_col3:
    st.markdown("<p class='plot-label'>Survival Month</p>", unsafe_allow_html=True)
    sel_survival = st.selectbox(
        "Survival Month",
        list(SURVIVAL_OPTIONS.keys()),
        index=5,   # default M6
        label_visibility="collapsed",
    )

surv_month_num = SURVIVAL_OPTIONS[sel_survival]
surv_plot_col  = f"surv_m{surv_month_num}"

# Join avg_gcv from LTV agg onto survival agg for x-axis
joined = surv_agg.merge(
    ltv_agg[["agent_dim", "product_dim", "customer_dim", "avg_gcv"]],
    on=["agent_dim", "product_dim", "customer_dim"],
    how="inner",
)

if joined.empty or surv_plot_col not in joined.columns:
    st.warning("No data available for this survival month with current filters.")
else:
    y_vals = joined[surv_plot_col].dropna() * 100
    y_min  = (int(max(0, y_vals.min() - 3)) // 5) * 5
    y_max  = min(102, ((int(y_vals.max() + 3) // 5) + 1) * 5)
    x_vals = joined["avg_gcv"].dropna()
    x_min  = max(0, x_vals.min() - 5)
    x_max  = x_vals.max() + 5

    fig3 = go.Figure()

    for val in surv_color_values:
        subset = joined[joined[color_col] == val]
        if subset.empty:
            continue
        hover_text = [
            f"<b>{r.agent_dim} · {r.product_dim} · {r.customer_dim}</b><br>"
            f"Avg GCV: ${r.avg_gcv:.2f}<br>"
            f"{sel_survival} Survival: {getattr(r, surv_plot_col) * 100:.1f}%<br>"
            f"Orders: {int(r.total_orders):,}"
            for r in subset.itertuples()
        ]
        fig3.add_trace(go.Scatter(
            x=subset["avg_gcv"],
            y=subset[surv_plot_col] * 100,
            mode="markers",
            name=str(val),
            marker=dict(color=surv_color_map[val], size=11, opacity=0.92, line=dict(width=0)),
            text=hover_text,
            hovertemplate="%{text}<extra></extra>",
        ))

    fig3.update_layout(
        **COMMON_LAYOUT,
        height=480,
        legend=make_legend(sel_color_by),
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
    st.plotly_chart(fig3, use_container_width=True)


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    f"<p style='font-family:DM Mono,monospace;font-size:0.65rem;color:#3a4060;"
    f"text-transform:uppercase;letter-spacing:0.08em;text-align:center;margin-top:2rem;'>"
    f"LTV: {len(ltv_raw):,} rows · Survival: {len(survival_raw):,} rows</p>",
    unsafe_allow_html=True,
)