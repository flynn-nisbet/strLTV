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
h2 {
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    color: #e8eaf0 !important;
    margin-top: 2rem !important;
    margin-bottom: 0.5rem !important;
}
h3 {
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    color: #e8eaf0 !important;
    margin-bottom: 0.25rem !important;
}
h4 {
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    color: #c4c8d8 !important;
    margin-top: 1.25rem !important;
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
/* Methodology tab prose styling */
.methodology-body p, .methodology-body li {
    font-family: 'Syne', sans-serif;
    font-size: 0.95rem;
    color: #c4c8d8;
    line-height: 1.75;
}
.methodology-body ul {
    padding-left: 1.5rem;
    margin-bottom: 1rem;
}
.methodology-body strong {
    color: #e8eaf0;
}
.methodology-body hr {
    border: none;
    border-top: 1px solid #252a3a;
    margin: 2rem 0;
}
.method-callout {
    background: #13161e;
    border: 1px solid #252a3a;
    border-left: 3px solid #5b8dee;
    border-radius: 6px;
    padding: 14px 18px;
    margin: 1rem 0 1.5rem 0;
    font-family: 'DM Mono', monospace;
    font-size: 0.78rem;
    color: #8b9bbf;
    line-height: 1.6;
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
    df = df[df["partner_name"] != "Atlantex Power"]
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
    df = df[df["partner_name"] != "Atlantex Power"]
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
# SIDEBAR  (always visible — controls dashboard tab only)
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

    def labeled_opts(df, col, prefix):
        return [f"{prefix} · {v}" for v in sorted(df[col].dropna().unique().tolist())]

    agent_opts = (
        labeled_opts(ltv_raw, "center_location",    "Center")
      + labeled_opts(ltv_raw, "conversion_quartile","Conv. Quartile")
      + labeled_opts(ltv_raw, "survival_quartile",  "Surv. Quartile")
      + labeled_opts(ltv_raw, "tenure_category",    "Tenure")
    )
    product_opts = (
        labeled_opts(ltv_raw, "sold_product_type", "Plan Type")
      + labeled_opts(ltv_raw, "partner_name",      "Provider")
    )
    customer_opts = (
        labeled_opts(ltv_raw, "mover_switcher",     "Mover/Switcher")
      + labeled_opts(ltv_raw, "brand_category",     "Brand")
      + labeled_opts(ltv_raw, "site_serp_category", "Site/SERP")
      + labeled_opts(ltv_raw, "usage_band",         "Usage Band")
      + labeled_opts(ltv_raw, "consistent_usage",   "Consistent Usage")
    )

    st.markdown("<p class='sidebar-section'>Filter Values</p>", unsafe_allow_html=True)

    st.markdown("<p class='sidebar-label'>Agent</p>", unsafe_allow_html=True)
    sel_agent_vals = st.multiselect(
        "Agent", agent_opts, default=[], placeholder="All agent values",
        label_visibility="collapsed",
    )

    st.markdown("<p class='sidebar-label'>Product</p>", unsafe_allow_html=True)
    sel_product_vals = st.multiselect(
        "Product", product_opts, default=[], placeholder="All product values",
        label_visibility="collapsed",
    )

    st.markdown("<p class='sidebar-label'>Customer</p>", unsafe_allow_html=True)
    sel_customer_vals = st.multiselect(
        "Customer", customer_opts, default=[], placeholder="All customer values",
        label_visibility="collapsed",
    )

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

def build_col_filter(labeled_selections, prefix_col_map):
    col_values = {}
    for item in labeled_selections:
        prefix, value = item.split(" · ", 1)
        col = prefix_col_map.get(prefix)
        if col:
            col_values.setdefault(col, set()).add(value)
    return col_values

AGENT_PREFIX_MAP = {
    "Center":          "center_location",
    "Conv. Quartile":  "conversion_quartile",
    "Surv. Quartile":  "survival_quartile",
    "Tenure":          "tenure_category",
}
PRODUCT_PREFIX_MAP = {
    "Plan Type": "sold_product_type",
    "Provider":  "partner_name",
}
CUSTOMER_PREFIX_MAP = {
    "Mover/Switcher":   "mover_switcher",
    "Brand":            "brand_category",
    "Site/SERP":        "site_serp_category",
    "Usage Band":       "usage_band",
    "Consistent Usage": "consistent_usage",
}

def apply_filters(df):
    d = df.copy()
    for col_filters in [
        build_col_filter(sel_agent_vals,    AGENT_PREFIX_MAP),
        build_col_filter(sel_product_vals,  PRODUCT_PREFIX_MAP),
        build_col_filter(sel_customer_vals, CUSTOMER_PREFIX_MAP),
    ]:
        for col, values in col_filters.items():
            if col in d.columns:
                d = d[d[col].isin(values)]
    return d

ltv      = apply_filters(ltv_raw)
survival = apply_filters(survival_raw)

# =============================================================================
# AGGREGATE
# =============================================================================

def aggregate_ltv(df, agent_col, product_col, customer_col):
    surv_means = {f"surv_m{m}": (f"survived_m{m}", "mean")
                  for m in range(1, 7) if f"survived_m{m}" in df.columns}
    agg_dict = {
        "avg_gcv":         ("gcv",                  "mean"),
        "avg_ltv":         ("derived_ltv",           "mean"),
        "avg_trailing":    ("trailing_revenue",      "mean"),
        "avg_upfront":     ("ltv_upfront_realized",  "mean"),
        "avg_gap":         ("gcv_ltv_gap",           "mean"),
        "total_orders":    ("gcv",                   "count"),
        "total_gcv":       ("gcv",                   "sum"),
        "total_ltv":       ("derived_ltv",           "sum"),
        "activation_rate": ("activated_ind",         "mean"),
        "first_pmt_rate":  ("first_payment_ind",     "mean"),
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
    grp["gcv_pct_rank"]   = grp["avg_gcv"].rank(pct=True).round(4)
    grp["ltv_pct_rank"]   = grp["avg_ltv"].rank(pct=True).round(4)
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

color_col    = COLOR_BY_OPTIONS[sel_color_by]
color_values = sorted(ltv_agg[color_col].dropna().unique().tolist()) \
               if color_col in ltv_agg.columns else []
color_map    = {v: PALETTES[i % len(PALETTES)] for i, v in enumerate(color_values)}

surv_color_values = sorted(surv_agg[color_col].dropna().unique().tolist()) \
                    if color_col in surv_agg.columns else []
surv_color_map    = {v: PALETTES[i % len(PALETTES)] for i, v in enumerate(surv_color_values)}

# =============================================================================
# TABS
# =============================================================================

tab_dashboard, tab_methodology, tab_takeaways = st.tabs([
    "📊  Dashboard", "📖  Methodology", "📋  Takeaways"
])

# =============================================================================
# TAB 1 — DASHBOARD
# =============================================================================

with tab_dashboard:

    st.markdown("# GCV vs LTV Pipeline")
    st.markdown(
        f"<p style='font-family:DM Mono,monospace;font-size:0.72rem;color:#6b7280;"
        f"letter-spacing:0.08em;text-transform:uppercase;margin-top:-12px;margin-bottom:8px;'>"
        f"{sel_agent_dim} × {sel_product_dim} × {sel_customer_dim} · "
        f"{len(ltv):,} LTV orders · {len(survival):,} survival orders</p>",
        unsafe_allow_html=True,
    )
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # ── KPI row ───────────────────────────────────────────────────────────────
    n_seg   = len(ltv_agg)
    avg_gcv = ltv_agg["avg_gcv"].mean() if n_seg > 0 else None
    avg_ltv = ltv_agg["avg_ltv"].mean() if n_seg > 0 else None
    avg_gap = ltv_agg["avg_gap"].mean() if n_seg > 0 else None
    avg_act = ltv["activated_ind"].mean() * 100 if len(ltv) > 0 else None

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

    # ── Section 1: Rank Scatter ───────────────────────────────────────────────
    st.markdown("<p class='section-header'>01 · Rank Comparison</p>", unsafe_allow_html=True)
    st.markdown("### GCV Percentile vs LTV Percentile")

    if n_seg == 0:
        st.warning("No data matches the selected filters.")
    else:
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines",
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
                x=subset["gcv_pct_rank"], y=subset["ltv_pct_rank"],
                mode="markers", name=str(val),
                marker=dict(color=color_map[val], size=11, opacity=0.92, line=dict(width=0)),
                text=hover_text, hovertemplate="%{text}<extra></extra>",
            ))
        fig1.update_layout(
            **COMMON_LAYOUT, height=480, legend=make_legend(sel_color_by),
            xaxis=dict(title=dict(text="GCV NORMALIZED RANK (0–1)",
                                  font=dict(size=11, color="#6b7280")),
                       range=[-0.02, 1.02], **AXIS_STYLE),
            yaxis=dict(title=dict(text="LTV NORMALIZED RANK (0–1)",
                                  font=dict(size=11, color="#6b7280")),
                       range=[-0.02, 1.02], **AXIS_STYLE),
            annotations=[dict(x=0.98, y=1.01, text="GCV = LTV", showarrow=False,
                               font=dict(size=10, color="#3a4060",
                                         family="DM Mono, monospace"),
                               xanchor="right")],
        )
        st.plotly_chart(fig1, use_container_width=True)

    # ── Section 2: Survival Curve Fan ─────────────────────────────────────────
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
            x=x_labels, y=overall_median, mode="lines",
            name="Overall Median",
            line=dict(color="#3a4060", width=2, dash="dot"),
            hovertemplate="Overall median: %{y:.1f}%<extra></extra>",
        ))
        for val in surv_color_values:
            subset  = surv_agg[surv_agg[color_col] == val]
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
                x=x_labels, y=curve, mode="lines+markers", name=str(val),
                line=dict(color=surv_color_map[val], width=2.5),
                marker=dict(color=surv_color_map[val], size=7),
                hovertemplate=f"<b>{val}</b><br>Month: %{{x}}<br>Survival: %{{y:.1f}}%<extra></extra>",
            ))
        fig2.update_layout(
            **COMMON_LAYOUT, height=420, legend=make_legend(sel_color_by),
            xaxis=dict(title=dict(text="MONTH", font=dict(size=11, color="#6b7280")),
                       **AXIS_STYLE),
            yaxis=dict(title=dict(text="SURVIVAL RATE (%)", font=dict(size=11, color="#6b7280")),
                       range=[0, 105], ticksuffix="%", **AXIS_STYLE),
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── Section 3: Survival vs GCV ────────────────────────────────────────────
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    st.markdown("<p class='section-header'>03 · Survival Analysis</p>", unsafe_allow_html=True)
    st.markdown("### Survival Rate vs Avg GCV")

    surv_col3, _ = st.columns([1, 2])
    with surv_col3:
        st.markdown("<p class='plot-label'>Survival Month</p>", unsafe_allow_html=True)
        sel_survival = st.selectbox(
            "Survival Month", list(SURVIVAL_OPTIONS.keys()), index=5,
            label_visibility="collapsed",
        )

    surv_month_num = SURVIVAL_OPTIONS[sel_survival]
    surv_plot_col  = f"surv_m{surv_month_num}"

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
                x=subset["avg_gcv"], y=subset[surv_plot_col] * 100,
                mode="markers", name=str(val),
                marker=dict(color=surv_color_map[val], size=11, opacity=0.92,
                            line=dict(width=0)),
                text=hover_text, hovertemplate="%{text}<extra></extra>",
            ))
        fig3.update_layout(
            **COMMON_LAYOUT, height=480, legend=make_legend(sel_color_by),
            xaxis=dict(title=dict(text="AVG GCV ($)", font=dict(size=11, color="#6b7280")),
                       range=[x_min, x_max], tickprefix="$", **AXIS_STYLE),
            yaxis=dict(title=dict(text=f"{sel_survival.upper()} SURVIVAL RATE (%)",
                                  font=dict(size=11, color="#6b7280")),
                       range=[y_min, y_max], ticksuffix="%", **AXIS_STYLE),
        )
        st.plotly_chart(fig3, use_container_width=True)

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown(
        f"<p style='font-family:DM Mono,monospace;font-size:0.65rem;color:#3a4060;"
        f"text-transform:uppercase;letter-spacing:0.08em;text-align:center;margin-top:2rem;'>"
        f"LTV: {len(ltv_raw):,} rows · Survival: {len(survival_raw):,} rows</p>",
        unsafe_allow_html=True,
    )

# =============================================================================
# TAB 2 — METHODOLOGY
# =============================================================================

with tab_methodology:

    METHODOLOGY_FILE = os.path.join(os.path.dirname(__file__), "pipeline_methodology.md")

    st.markdown("# GCV vs LTV Pipeline — Methodology")
    st.markdown(
        "<p style='font-family:DM Mono,monospace;font-size:0.72rem;color:#6b7280;"
        "letter-spacing:0.08em;text-transform:uppercase;margin-top:-12px;margin-bottom:8px;'>"
        "How the analysis works</p>",
        unsafe_allow_html=True,
    )
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    if os.path.exists(METHODOLOGY_FILE):
        with open(METHODOLOGY_FILE, "r") as f:
            methodology_md = f.read()

        # Render section by section so we can inject callout boxes
        # between the major sections for quick-reference highlights
        sections = methodology_md.split("\n---\n")

        callouts = [
            None,  # Overview — no callout
            "<div class='method-callout'>📅 <b>Survival cohort:</b> Sep–Dec 2024 · 12-month flags · historical benchmark<br>📅 <b>LTV cohort:</b> Mar–Jul 2025 · 6-month cap · current order valuation</div>",
            "<div class='method-callout'>🏠 Rentcast enrichment → 🌡️ Weather features → 💬 Call survey inputs → ⚙️ Feature engineering → 🤖 LightGBM scoring → kWh per month per call</div>",
            "<div class='method-callout'>💰 <b>Derived LTV</b> = Upfront bounty (where earned) + Σ(residual_mN × mil_rate / 1000 × survived_mN) for N in 1–6</div>",
            None,  # Agent quartiles — no callout
            None,  # Key design choices — no callout
        ]

        for i, section in enumerate(sections):
            st.markdown(
                f"<div class='methodology-body'>{section}</div>",
                unsafe_allow_html=True,
            )
            if i < len(callouts) and callouts[i]:
                st.markdown(callouts[i], unsafe_allow_html=True)

    else:
        st.info(
            "Methodology file `pipeline_methodology.md` not found in the app directory. "
            "Place it alongside `app.py` to render it here.",
            icon="📄",
        )

# =============================================================================
# TAB 3 — TAKEAWAYS
# =============================================================================

# =============================================================================
# TAB 3 — TAKEAWAYS
# =============================================================================

with tab_takeaways:

    st.markdown("# Key Takeaways")
    st.markdown(
        "<p style='font-family:DM Mono,monospace;font-size:0.72rem;color:#6b7280;"
        "letter-spacing:0.08em;text-transform:uppercase;margin-top:-12px;margin-bottom:8px;'>"
        "Where static LTV assumptions diverge most from dynamic model output</p>",
        unsafe_allow_html=True,
    )
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    st.markdown(
        "<p class='section-header'>Proof Point — Similar GCV, Divergent LTV</p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "### Segments with Similar GCV Rank but Dramatically Different LTV Rank"
    )
    st.markdown(
        "<p style='font-family:DM Mono,monospace;font-size:0.68rem;color:#6b7280;"
        "margin-top:-8px;margin-bottom:16px;'>"
        "Computed across every agent × product × customer dimension combination. "
        "Segments with at least 100 orders, ordered by largest rank divergence. "
        "Segments a static model would treat as equivalent that the dynamic model "
        "separates most appear at the top."
        "</p>",
        unsafe_allow_html=True,
    )

    # ── Build aggregation across ALL dimension combinations ───────────────────
    all_agent_cols    = list(AGENT_DIMS.values())
    all_product_cols  = list(PRODUCT_DIMS.values())
    all_customer_cols = list(CUSTOMER_DIMS.values())

    agent_label_map    = {v: k for k, v in AGENT_DIMS.items()}
    product_label_map  = {v: k for k, v in PRODUCT_DIMS.items()}
    customer_label_map = {v: k for k, v in CUSTOMER_DIMS.items()}

    combo_frames = []

    for a_col in all_agent_cols:
        for p_col in all_product_cols:
            for c_col in all_customer_cols:

                ltv_has = all(c in ltv_raw.columns for c in [a_col, p_col, c_col])
                if not ltv_has:
                    continue

                grp = (
                    ltv_raw
                    .dropna(subset=[a_col, p_col, c_col])
                    .groupby([a_col, p_col, c_col])
                    .agg(
                        total_orders    = ("gcv",                  "count"),
                        avg_gcv         = ("gcv",                  "mean"),
                        avg_ltv         = ("derived_ltv",          "mean"),
                        avg_upfront     = ("ltv_upfront_realized", "mean"),
                        avg_trailing    = ("trailing_revenue",     "mean"),
                        avg_gap         = ("gcv_ltv_gap",          "mean"),
                        activation_rate = ("activated_ind",        "mean"),
                        first_pmt_rate  = ("first_payment_ind",    "mean"),
                        surv_m1         = ("survived_m1",          "mean"),
                        surv_m2         = ("survived_m2",          "mean"),
                        surv_m3         = ("survived_m3",          "mean"),
                        surv_m4         = ("survived_m4",          "mean"),
                        surv_m5         = ("survived_m5",          "mean"),
                        surv_m6         = ("survived_m6",          "mean"),
                    )
                    .reset_index()
                    .rename(columns={
                        a_col: "dim_agent",
                        p_col: "dim_product",
                        c_col: "dim_customer",
                    })
                )

                grp["agent_field"]    = a_col
                grp["product_field"]  = p_col
                grp["customer_field"] = c_col
                grp["combo_label"]    = (
                    f"{agent_label_map.get(a_col, a_col)} × "
                    f"{product_label_map.get(p_col, p_col)} × "
                    f"{customer_label_map.get(c_col, c_col)}"
                )

                combo_frames.append(grp)

    if not combo_frames:
        st.warning("Could not build any dimension combinations from the loaded data.")
    else:
        proof_agg = pd.concat(combo_frames, ignore_index=True)

        # ── Volume filter ─────────────────────────────────────────────────────
        proof_agg = proof_agg[proof_agg["total_orders"] >= 100].copy()

        # ── Percentile ranks across full stacked population ───────────────────
        proof_agg["gcv_pct_rank"]   = proof_agg["avg_gcv"].rank(pct=True).round(4)
        proof_agg["ltv_pct_rank"]   = proof_agg["avg_ltv"].rank(pct=True).round(4)
        proof_agg["rank_delta"]     = (proof_agg["gcv_pct_rank"] - proof_agg["ltv_pct_rank"]).round(4)
        proof_agg["rank_delta_abs"] = proof_agg["rank_delta"].abs()

        proof_agg["seg_label"] = (
            proof_agg["dim_agent"].astype(str) + " · " +
            proof_agg["dim_product"].astype(str) + " · " +
            proof_agg["dim_customer"].astype(str)
        )
        proof_agg["seg_key"] = (
            proof_agg["agent_field"].astype(str) + "||" +
            proof_agg["dim_agent"].astype(str) + "||" +
            proof_agg["product_field"].astype(str) + "||" +
            proof_agg["dim_product"].astype(str) + "||" +
            proof_agg["customer_field"].astype(str) + "||" +
            proof_agg["dim_customer"].astype(str)
        )

        # ── All segments ordered by rank delta, top 5 for scatter highlight ───
        all_divergers = proof_agg.sort_values("rank_delta_abs", ascending=False).copy()
        top_divergers = all_divergers.head(5).copy()

        if all_divergers.empty:
            st.warning(
                "No segments found with ≥100 orders. "
                "Check that both CSVs are loaded correctly."
            )
        else:
            # ── Callout ───────────────────────────────────────────────────────
            best = all_divergers.iloc[0]

            st.markdown(
                f"<div class='method-callout'>"
                f"📍 <b>Largest divergence found:</b> "
                f"<b>{best['combo_label']}</b><br>"
                f"<b>{best['dim_agent']} · {best['dim_product']} · {best['dim_customer']}</b><br>"
                f"GCV rank <b>{best['gcv_pct_rank']:.2f}</b> → "
                f"LTV rank <b>{best['ltv_pct_rank']:.2f}</b> "
                f"(Δ <b>{best['rank_delta']:+.2f}</b>) &nbsp;·&nbsp; "
                f"Avg GCV <b>${best['avg_gcv']:,.0f}</b> → "
                f"Avg LTV <b>${best['avg_ltv']:,.0f}</b><br>"
                f"Activation: <b>{best['activation_rate'] * 100:.1f}%</b> &nbsp;·&nbsp; "
                f"M6 Survival: <b>{best['surv_m6'] * 100:.1f}%</b> &nbsp;·&nbsp; "
                f"Avg Trailing Rev: <b>${best['avg_trailing']:,.0f}</b> &nbsp;·&nbsp; "
                f"Orders: <b>{int(best['total_orders']):,}</b>"
                f"</div>",
                unsafe_allow_html=True,
            )

            # ── Full scrollable table ─────────────────────────────────────────
            table = all_divergers[[
                "combo_label",
                "dim_agent", "dim_product", "dim_customer",
                "total_orders",
                "avg_gcv", "avg_ltv",
                "gcv_pct_rank", "ltv_pct_rank", "rank_delta",
            ]].copy().reset_index(drop=True)

            table["avg_gcv"]      = table["avg_gcv"].apply(lambda x: f"${x:,.0f}")
            table["avg_ltv"]      = table["avg_ltv"].apply(lambda x: f"${x:,.0f}")
            table["gcv_pct_rank"] = table["gcv_pct_rank"].apply(lambda x: f"{x:.3f}")
            table["ltv_pct_rank"] = table["ltv_pct_rank"].apply(lambda x: f"{x:.3f}")
            table["rank_delta"]   = table["rank_delta"].apply(lambda x: f"{x:+.3f}")

            table = table.rename(columns={
                "combo_label":  "Grouping",
                "dim_agent":    "Agent Value",
                "dim_product":  "Product Value",
                "dim_customer": "Customer Value",
                "total_orders": "Orders",
                "avg_gcv":      "Avg GCV",
                "avg_ltv":      "Avg LTV",
                "gcv_pct_rank": "GCV Rank",
                "ltv_pct_rank": "LTV Rank",
                "rank_delta":   "Rank Δ",
            })

            def color_rank_delta(val):
                try:
                    v = float(val)
                    if v > 0.15:
                        return "color: #e05c8a; font-weight: bold"
                    elif v < -0.15:
                        return "color: #3dd68c; font-weight: bold"
                    return "color: #e8eaf0"
                except Exception:
                    return ""

            styled = (
                table.style
                .map(color_rank_delta, subset=["Rank Δ"])
                .set_properties(**{
                    "font-family": "DM Mono, monospace",
                    "font-size":   "0.82rem",
                })
            )

            st.dataframe(styled, use_container_width=True, hide_index=True,
                         height=420)

            # ── Scatter ───────────────────────────────────────────────────────
            st.markdown("<div style='margin-top:2rem;'></div>", unsafe_allow_html=True)
            st.markdown(
                "<p class='section-header'>All Segments — Rank Divergence Map</p>",
                unsafe_allow_html=True,
            )
            st.markdown("### GCV Rank vs LTV Rank · All Dimension Combinations")
            st.markdown(
                "<p style='font-family:DM Mono,monospace;font-size:0.68rem;color:#6b7280;"
                "margin-top:-8px;margin-bottom:16px;'>"
                "Points on the diagonal satisfy the static model assumption. "
                "Orange highlighted points are the top 5 divergers.</p>",
                unsafe_allow_html=True,
            )

            highlight_keys = set(top_divergers["seg_key"])
            proof_agg["is_highlight"] = proof_agg["seg_key"].isin(highlight_keys)

            bg = proof_agg[~proof_agg["is_highlight"]].copy()
            hi = proof_agg[proof_agg["is_highlight"]].copy()

            fig_proof = go.Figure()

            fig_proof.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1], mode="lines",
                line=dict(color="#3a4060", width=1.5, dash="dash"),
                hoverinfo="skip", showlegend=False,
            ))

            fig_proof.add_trace(go.Scatter(
                x=bg["gcv_pct_rank"],
                y=bg["ltv_pct_rank"],
                mode="markers",
                name="All other segments",
                marker=dict(color="#2a3050", size=8, opacity=0.7, line=dict(width=0)),
                text=[
                    f"<b>{r.seg_label}</b><br>"
                    f"Grouping: {r.combo_label}<br>"
                    f"GCV Rank: {r.gcv_pct_rank:.3f}<br>"
                    f"LTV Rank: {r.ltv_pct_rank:.3f}<br>"
                    f"Rank Δ: {r.rank_delta:+.3f}<br>"
                    f"Avg GCV: ${r.avg_gcv:,.0f}<br>"
                    f"Avg LTV: ${r.avg_ltv:,.0f}<br>"
                    f"Activation: {r.activation_rate * 100:.1f}%<br>"
                    f"M6 Survival: {r.surv_m6 * 100:.1f}%<br>"
                    f"Orders: {int(r.total_orders):,}"
                    for r in bg.itertuples()
                ],
                hovertemplate="%{text}<extra></extra>",
            ))

            fig_proof.add_trace(go.Scatter(
                x=hi["gcv_pct_rank"],
                y=hi["ltv_pct_rank"],
                mode="markers+text",
                name="Top 5 divergers",
                marker=dict(
                    color="#f59e42", size=14, opacity=1.0,
                    line=dict(color="#e8eaf0", width=1.5),
                ),
                text=hi["seg_label"],
                textposition="top center",
                textfont=dict(family="DM Mono, monospace", size=9, color="#e8eaf0"),
                customdata=list(zip(
                    hi["rank_delta"].round(3),
                    (hi["surv_m6"] * 100).round(1),
                    (hi["activation_rate"] * 100).round(1),
                    hi["total_orders"].astype(int),
                    hi["avg_gcv"].round(0),
                    hi["avg_ltv"].round(0),
                    hi["avg_trailing"].round(0),
                    hi["combo_label"],
                )),
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "Grouping: %{customdata[7]}<br>"
                    "GCV Rank: %{x:.3f} → LTV Rank: %{y:.3f}<br>"
                    "Rank Δ: %{customdata[0]:+.3f}<br>"
                    "Avg GCV: $%{customdata[4]:,.0f} → Avg LTV: $%{customdata[5]:,.0f}<br>"
                    "Avg Trailing Rev: $%{customdata[6]:,.0f}<br>"
                    "M6 Survival: %{customdata[1]:.1f}%<br>"
                    "Activation: %{customdata[2]:.1f}%<br>"
                    "Orders: %{customdata[3]:,}"
                    "<extra></extra>"
                ),
            ))

            fig_proof.update_layout(
                **COMMON_LAYOUT,
                height=520,
                legend=make_legend("Segment"),
                xaxis=dict(
                    title=dict(text="GCV NORMALIZED RANK (0–1)",
                               font=dict(size=11, color="#6b7280")),
                    range=[-0.02, 1.18], **AXIS_STYLE,
                ),
                yaxis=dict(
                    title=dict(text="LTV NORMALIZED RANK (0–1)",
                               font=dict(size=11, color="#6b7280")),
                    range=[-0.02, 1.02], **AXIS_STYLE,
                ),
                annotations=[dict(
                    x=0.98, y=1.01,
                    text="Static assumption holds here",
                    showarrow=False,
                    font=dict(size=10, color="#3a4060", family="DM Mono, monospace"),
                    xanchor="right",
                )],
            )
            st.plotly_chart(fig_proof, use_container_width=True)

            # ── Footer ────────────────────────────────────────────────────────
            st.markdown(
                "<p style='font-family:DM Mono,monospace;font-size:0.65rem;color:#3a4060;"
                "text-transform:uppercase;letter-spacing:0.08em;margin-top:1.5rem;'>"
                "Survival from LTV cohort (Mar–Jul 2025, 6-month cap). "
                "Segments with &lt;100 orders excluded. "
                "Ranks computed across all dimension combinations stacked together."
                "</p>",
                unsafe_allow_html=True,
            )