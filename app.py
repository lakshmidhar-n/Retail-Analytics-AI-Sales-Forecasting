"""
MERIDIAN | Retail Intelligence Observatory
Streamlit dashboard: briefing, pulse, mix, stores, forecast, discount lab, segments.
Run:  streamlit run app.py
"""
import re
import warnings
from pathlib import Path
from io import BytesIO

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Meridian | Retail Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ───────────────────────── PALETTE ─────────────────────────
TEAL, AMBER, VIOLET = "#2DD4BF", "#F5A524", "#A78BFA"
ROSE, BLUE, GREEN = "#FB7185", "#60A5FA", "#34D399"
SEQ = [TEAL, AMBER, VIOLET, BLUE, ROSE, GREEN, "#F472B6", "#94A3B8"]
MUTED, TEXT, GRID = "#8EA0BC", "#E6EDF7", "rgba(148,163,184,0.12)"

# ───────────────────────── THEME / CSS ─────────────────────────
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
html, body, [class*="css"], .stApp {font-family:'Inter',system-ui,sans-serif;}
.stApp {background: radial-gradient(1200px 600px at 85% -10%, rgba(45,212,191,.08), transparent 60%),
        radial-gradient(900px 500px at 0% 0%, rgba(167,139,250,.07), transparent 55%), #070B14;}
#MainMenu, footer, .stDeployButton, [data-testid="stToolbar"], [data-testid="stDecoration"] {display:none !important;}
header[data-testid="stHeader"] {background:transparent;}
.block-container {padding-top:1.4rem; padding-bottom:3rem; max-width:1500px;}
h1,h2,h3 {font-family:'Fraunces',Georgia,serif !important; letter-spacing:-.01em; font-weight:600 !important;}

/* sidebar */
[data-testid="stSidebar"] {background:#090E19; border-right:1px solid #16203A;}
[data-testid="stSidebar"] div[role="radiogroup"] {gap:2px;}
[data-testid="stSidebar"] label[data-baseweb="radio"] {padding:9px 12px; border-radius:9px; width:100%; margin:0; transition:.15s;}
[data-testid="stSidebar"] label[data-baseweb="radio"] > div:first-child {display:none;}
[data-testid="stSidebar"] label[data-baseweb="radio"]:hover {background:#101A30;}
[data-testid="stSidebar"] label[data-baseweb="radio"]:has(input:checked) {background:linear-gradient(90deg, rgba(45,212,191,.16), rgba(45,212,191,.02)); box-shadow: inset 2px 0 0 #2DD4BF;}
[data-testid="stSidebar"] label[data-baseweb="radio"] p {font-size:14px; font-weight:500; color:#C9D6EA;}
.brand {font-family:'Fraunces',serif; font-size:26px; font-weight:600; color:#fff; line-height:1;}
.brand-sub {font-family:'JetBrains Mono',monospace; font-size:10px; letter-spacing:.18em; color:#6B7C99; margin-top:6px; text-transform:uppercase;}
.side-label {font-family:'JetBrains Mono',monospace; font-size:10px; letter-spacing:.16em; color:#586784; text-transform:uppercase; margin:18px 0 6px;}

/* hero */
.eyebrow {font-family:'JetBrains Mono',monospace; font-size:11px; letter-spacing:.2em; color:#F5A524; text-transform:uppercase;}
.hero-h {font-family:'Fraunces',Georgia,serif; font-size:clamp(30px,4vw,50px); line-height:1.08; color:#F4F7FC; margin:10px 0 10px; font-weight:400; letter-spacing:-.02em;}
.hero-h em {color:#F5A524; font-style:italic;}
.hero-p {color:#8EA0BC; font-size:15px; max-width:820px; line-height:1.55;}
.page-h {font-family:'Fraunces',Georgia,serif; font-size:34px; color:#F4F7FC; margin:4px 0 2px; font-weight:400; letter-spacing:-.015em;}
.page-h em {color:#F5A524;}
.page-p {color:#8EA0BC; font-size:14px; margin-bottom:14px;}

/* kpi */
.kpi-grid {display:grid; grid-template-columns:repeat(6,minmax(0,1fr)); gap:12px; margin:18px 0 10px;}
.kpi {background:linear-gradient(180deg,#0F1729,#0B1222); border:1px solid #1A2540; border-radius:14px; padding:14px 16px 10px; position:relative; overflow:hidden;}
.kpi::before {content:""; position:absolute; left:0; top:0; bottom:0; width:3px; background:var(--c,#2DD4BF);}
.kpi-l {font-family:'JetBrains Mono',monospace; font-size:10px; letter-spacing:.14em; color:#7C8DAA; text-transform:uppercase;}
.kpi-v {font-size:26px; font-weight:600; color:#fff; margin:6px 0 2px; letter-spacing:-.02em;}
.kpi-d {font-size:12px; font-weight:500;}
.kpi-d.up {color:#34D399;} .kpi-d.dn {color:#FB7185;} .kpi-d.na {color:#6B7C99;}
.kpi svg {width:100%; height:30px; margin-top:6px; display:block;}
@media (max-width:1100px){.kpi-grid{grid-template-columns:repeat(3,minmax(0,1fr));}}
@media (max-width:640px){.kpi-grid{grid-template-columns:repeat(2,minmax(0,1fr));}}

/* brief cards */
.brief-grid {display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:12px; margin:8px 0 16px;}
.brief {background:#0D1426; border:1px solid #1A2540; border-top:2px solid var(--c); border-radius:12px; padding:14px 16px;}
.brief-t {font-family:'JetBrains Mono',monospace; font-size:10px; letter-spacing:.16em; color:var(--c); text-transform:uppercase;}
.brief-h {font-size:15.5px; font-weight:600; color:#F1F5FB; margin:7px 0 5px; line-height:1.3;}
.brief-b {font-size:13px; color:#8EA0BC; line-height:1.5;}
@media (max-width:1000px){.brief-grid{grid-template-columns:1fr;}}
.sec {font-family:'JetBrains Mono',monospace; font-size:11px; letter-spacing:.18em; color:#586784; text-transform:uppercase; margin:22px 0 8px;}

/* card titles / containers */
.ct {font-size:15px; font-weight:600; color:#EAF0FA; margin-bottom:2px;}
.cs {font-size:12px; color:#7C8DAA; margin-bottom:6px;}
[data-testid="stVerticalBlockBorderWrapper"] {background:#0C1325; border-color:#1A2540 !important; border-radius:14px;}

/* ask box */
.chip-note {font-size:12px; color:#6B7C99; margin:4px 0;}
.answer {background:#0D1426; border:1px solid #1A2540; border-left:3px solid #F5A524; border-radius:12px; padding:16px 20px; margin:10px 0;}
.answer .a-l {font-family:'JetBrains Mono',monospace; font-size:10px; letter-spacing:.16em; color:#7C8DAA; text-transform:uppercase;}
.answer .a-v {font-family:'Fraunces',serif; font-size:30px; color:#fff; margin:4px 0;}
.answer .a-s {color:#8EA0BC; font-size:13px;}
.bar-row {display:grid; grid-template-columns:150px 1fr 110px; gap:12px; align-items:center; margin:7px 0; font-size:13px;}
.bar-row .n {color:#C9D6EA; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;}
.bar-row .b {height:9px; border-radius:6px; background:#121C33; overflow:hidden;}
.bar-row .b i {display:block; height:100%; border-radius:6px; background:linear-gradient(90deg,#F5A524,#FBBF5E);}
.bar-row .v {text-align:right; color:#E6EDF7; font-family:'JetBrains Mono',monospace; font-size:12px;}

/* widgets */
span[data-baseweb="tag"] {background:rgba(45,212,191,.13) !important; border:1px solid rgba(45,212,191,.35); border-radius:7px !important;}
span[data-baseweb="tag"] span {color:#7EF0DF !important;}
div[data-baseweb="select"] > div, .stTextInput input, .stDateInput input {background:#0E1527 !important; border-color:#1E2B4A !important; border-radius:10px !important;}
.stButton > button {background:#0E1527; border:1px solid #22304F; color:#C9D6EA; border-radius:999px; font-size:12px; padding:2px 12px;}
.stButton > button:hover {border-color:#2DD4BF; color:#2DD4BF;}
.stButton > button[kind="primary"] {background:#F5A524; color:#1a1204; border:none; font-weight:600; border-radius:10px;}
.stDownloadButton > button {background:#0E1527; border:1px solid #22304F; color:#C9D6EA; border-radius:10px;}
[data-testid="stMetric"] {background:#0F1729; border:1px solid #1A2540; border-radius:12px; padding:12px 14px;}
[data-testid="stMetricLabel"] p {font-family:'JetBrains Mono',monospace; font-size:10px; letter-spacing:.12em; text-transform:uppercase; color:#7C8DAA;}
[data-testid="stDataFrame"] {border:1px solid #1A2540; border-radius:12px; overflow:hidden;}
.foot {color:#586784; font-size:12px; border-top:1px solid #16203A; margin-top:34px; padding-top:14px;}
.pill {display:inline-block; padding:2px 9px; border-radius:999px; font-size:11px; font-family:'JetBrains Mono',monospace; border:1px solid #22304F; color:#8EA0BC; margin-right:6px;}
"""
st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)


# ───────────────────────── HELPERS ─────────────────────────
def html(s: str):
    st.markdown("".join(line.strip() for line in s.splitlines()), unsafe_allow_html=True)


def show(fig, key=None):
    try:
        st.plotly_chart(fig, width="stretch", key=key, config={"displayModeBar": False})
    except TypeError:
        st.plotly_chart(fig, use_container_width=True, key=key, config={"displayModeBar": False})


def table(df_, **kw):
    try:
        st.dataframe(df_, width="stretch", hide_index=True, **kw)
    except TypeError:
        st.dataframe(df_, use_container_width=True, hide_index=True, **kw)


def style(fig, h=340, legend=False, ytitle=None):
    fig.update_layout(
        height=h, margin=dict(l=6, r=6, t=10, b=6),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#C9D6EA", size=12),
        showlegend=legend, colorway=SEQ,
        legend=dict(orientation="h", y=1.12, x=0, font=dict(color="#C9D6EA"), bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor="#0B1222", bordercolor="#2DD4BF", font=dict(color="#fff", size=12)),
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor="#1A2540", tickfont=dict(color=MUTED), title=None)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor="#1A2540", tickfont=dict(color=MUTED), title=ytitle)
    return fig


def card(title, sub=None):
    html(f'<div class="ct">{title}</div>' + (f'<div class="cs">{sub}</div>' if sub else ""))


def cr(v):
    v = v / 1e7
    return f"₹{v:,.0f} Cr" if abs(v) >= 100 else f"₹{v:,.2f} Cr"


def lakh(v):
    return f"₹{v / 1e5:,.2f} L"


def num(v):
    return f"{v:,.0f}"


def pct(v, d=1):
    return "—" if v is None or pd.isna(v) or np.isinf(v) else f"{v:+.{d}f}%"


def spark(vals, color):
    vals = np.asarray(vals, dtype=float)
    if len(vals) < 2 or np.nanmax(vals) == np.nanmin(vals):
        return ""
    x = np.linspace(0, 100, len(vals))
    y = 26 - (vals - vals.min()) / (vals.max() - vals.min()) * 22
    pts = " ".join(f"{a:.1f},{b:.1f}" for a, b in zip(x, y))
    return (f'<svg viewBox="0 0 100 30" preserveAspectRatio="none"><polygon points="0,30 {pts} 100,30" fill="{color}" opacity=".12"/>'
            f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="1.6" vector-effect="non-scaling-stroke"/></svg>')


def kpi(label, value, delta, series, color, good_up=True):
    if delta is None or pd.isna(delta):
        d_html = '<div class="kpi-d na">no prior period</div>'
    else:
        up = delta >= 0
        good = up == good_up
        d_html = f'<div class="kpi-d {"up" if good else "dn"}">{"▲" if up else "▼"} {abs(delta):.1f}% vs prior year</div>'
    return (f'<div class="kpi" style="--c:{color}"><div class="kpi-l">{label}</div><div class="kpi-v">{value}</div>'
            f'{d_html}{spark(series, color)}</div>')


def bars_html(labels, values, fmt):
    mx = max(values) if len(values) and max(values) > 0 else 1
    rows = "".join(
        f'<div class="bar-row"><div class="n">{l}</div><div class="b"><i style="width:{max(v / mx * 100, 1):.1f}%"></i></div><div class="v">{fmt(v)}</div></div>'
        for l, v in zip(labels, values))
    return rows


# ───────────────────────── DATA ─────────────────────────
@st.cache_data(show_spinner="Loading retail data…")
def load_data(file_bytes=None):
    if file_bytes is not None:
        raw = pd.read_csv(BytesIO(file_bytes))
    else:
        here = Path(__file__).parent
        cands = sorted(list(here.glob("Retail_Sales_Data*.csv")) + list(Path.cwd().glob("Retail_Sales_Data*.csv")))
        if not cands:
            cands = sorted(list(here.glob("*.csv")) + list(Path.cwd().glob("*.csv")))
        if not cands:
            return None
        raw = pd.read_csv(cands[0])

    need = ["Date", "Store_ID", "Store_Location", "Product_Category", "Revenue", "Total_Sales", "Units_Sold", "Region"]
    miss = [c for c in need if c not in raw.columns]
    if miss:
        raise ValueError("Dataset is missing required columns: " + ", ".join(miss))

    raw["Date"] = pd.to_datetime(raw["Date"], errors="coerce")
    raw = raw.dropna(subset=["Date", "Revenue"]).copy()
    for c in ["Revenue", "Total_Sales", "Units_Sold", "Unit_Price", "Discount_Percentage", "Stock_On_Hand", "Store_Rating", "Holiday_Flag"]:
        if c in raw.columns:
            raw[c] = pd.to_numeric(raw[c], errors="coerce")
    for c in raw.columns:
        if raw[c].dtype == object or str(raw[c].dtype).startswith(("str", "string")):
            raw[c] = raw[c].astype(str).str.strip()
    for c, default in [("Brand", "Unknown"), ("Customer_Type", "Unknown"), ("Payment_Mode", "Unknown"),
                       ("Promotion_Applied", "No"), ("Product_Subcategory", "Unknown")]:
        if c not in raw.columns:
            raw[c] = default
    for c in ["Discount_Percentage", "Store_Rating", "Holiday_Flag"]:
        if c not in raw.columns:
            raw[c] = 0
    raw["Year"] = raw["Date"].dt.year
    raw["Month"] = raw["Date"].dt.to_period("M").dt.to_timestamp()
    raw["Discount_Amount"] = raw["Total_Sales"] - raw["Revenue"]
    return raw.sort_values("Date").reset_index(drop=True)


# ───────────────────────── SIDEBAR ─────────────────────────
with st.sidebar:
    html('<div class="brand">Meridian</div><div class="brand-sub">Retail Intelligence Observatory</div>')
    html('<div class="side-label">Navigate</div>')
    PAGES = ["Briefing", "Pulse", "Mix", "Stores", "Forecast", "Discount Lab", "Segments", "Data & Methods"]
    page = st.radio("Navigate", PAGES, label_visibility="collapsed")
    html('<div class="side-label">Data source</div>')
    uploaded = st.file_uploader("Upload a retail CSV (optional)", type=["csv"], label_visibility="collapsed")

try:
    df = load_data(uploaded.getvalue()) if uploaded else load_data()
except Exception as exc:
    st.error(f"Could not read the dataset: {exc}")
    st.stop()
if df is None:
    st.warning("No CSV found. Put `Retail_Sales_Data_Unlox.csv` next to app.py, or upload one from the sidebar.")
    st.stop()

# ───────────────────────── GLOBAL FILTER BAR ─────────────────────────
dmin, dmax = df["Date"].min().date(), df["Date"].max().date()
f1, f2, f3, f4 = st.columns([1.15, 1, 1.25, 1.25])
with f1:
    drange = st.date_input("Date range", (dmin, dmax), min_value=dmin, max_value=dmax)
with f2:
    sel_reg = st.multiselect("Region", sorted(df["Region"].unique()), default=sorted(df["Region"].unique()))
with f3:
    sel_sto = st.multiselect("Store", sorted(df["Store_Location"].unique()), default=sorted(df["Store_Location"].unique()))
with f4:
    sel_cat = st.multiselect("Category", sorted(df["Product_Category"].unique()), default=sorted(df["Product_Category"].unique()))

if isinstance(drange, (tuple, list)) and len(drange) == 2:
    d0, d1 = drange
else:
    d0, d1 = dmin, dmax
d0, d1 = pd.Timestamp(d0), pd.Timestamp(d1)

mask_nodate = (df["Region"].isin(sel_reg) & df["Store_Location"].isin(sel_sto) & df["Product_Category"].isin(sel_cat))
base = df[mask_nodate]
F = base[(base["Date"] >= d0) & (base["Date"] <= d1)]
if F.empty:
    st.warning("No rows match these filters. Widen the selection above.")
    st.stop()

# prior-year comparison window
prev0, prev1 = d0 - pd.DateOffset(years=1), d1 - pd.DateOffset(years=1)
P = base[(base["Date"] >= prev0) & (base["Date"] <= prev1)]
has_prev = len(P) > 0 and prev0 >= df["Date"].min() - pd.Timedelta(days=1)

with st.sidebar:
    html('<div class="side-label">Scope</div>')
    html(f'<span class="pill">{len(F):,} rows</span><span class="pill">{F["Store_Location"].nunique()} stores</span>'
         f'<div style="color:#6B7C99;font-size:12px;margin-top:8px">{F["Date"].min():%d %b %Y} → {F["Date"].max():%d %b %Y}</div>')
    st.download_button("⬇ Export filtered data", F.to_csv(index=False).encode("utf-8"),
                       "filtered_retail_data.csv", "text/csv")

# aggregates
rev, gross, units, tx = F["Revenue"].sum(), F["Total_Sales"].sum(), F["Units_Sold"].sum(), len(F)
disc = F["Discount_Amount"].sum()
ticket = rev / tx
n_months = F["Month"].nunique()


def yoy(cur, prev):
    return (cur / prev - 1) * 100 if has_prev and prev else None


# shared helpers that use F -----------------------------------------------------
def monthly(col="Revenue", how="sum", data=None):
    d = F if data is None else data
    return d.groupby("Month")[col].agg(how)


def anova_p(groups):
    groups = [g for g in groups if len(g) > 1]
    return stats.f_oneway(*groups).pvalue if len(groups) > 1 else np.nan


# ═════════════════════════ PAGE: BRIEFING ═════════════════════════
if page == "Briefing":
    html(f'<div class="eyebrow">Meridian · Retail Intelligence Observatory</div>'
         f'<div class="hero-h">{cr(rev)} in {n_months} months across {F["Store_Location"].nunique()} stores, '
         f'and <em>what actually drives it.</em></div>'
         f'<div class="hero-p">{tx:,} transactions from {F["Date"].min():%d %b %Y} to {F["Date"].max():%d %b %Y}. '
         f'Every figure below recalculates from your filters. Ask a question in plain English, or open a section on the left.</div>')

    # ---- ask the data
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    if "ask" not in st.session_state:
        st.session_state.ask = ""
    qc, bc = st.columns([6, 1])
    q = qc.text_input("Ask the data", key="ask", placeholder='e.g. "best store in 2024 by revenue", "revenue of Fashion in 2023"',
                      label_visibility="collapsed")
    bc.button("Ask", type="primary")
    examples = ["best store in 2024", "revenue of Fashion in 2023", "which category sells most units", "worst month by revenue",
                "average ticket by store", "how much discount did we give by category", "returning vs new customers"]
    ec = st.columns(len(examples))
    for i, ex in enumerate(examples):
        if ec[i].button(ex, key=f"ex{i}"):
            st.session_state.ask = ex
            st.rerun()

    def answer(query):
        ql = query.lower()
        d = F.copy()
        scope = []
        for col in ["Product_Category", "Region", "Store_Location", "Brand", "Payment_Mode"]:
            for v in sorted(df[col].unique(), key=len, reverse=True):
                if re.search(rf"\b{re.escape(v.lower())}\b", ql):
                    d = d[d[col] == v]
                    scope.append(v)
                    break
        for y in sorted(df["Year"].unique()):
            if str(y) in ql:
                d = d[d["Year"] == y]
                scope.append(str(y))
        if "returning" in ql and "new" not in ql:
            d = d[d["Customer_Type"] == "Returning"]; scope.append("Returning customers")
        elif re.search(r"\bnew (customer|buyer)", ql) and "returning" not in ql:
            d = d[d["Customer_Type"] == "New"]; scope.append("New customers")
        if d.empty:
            return None

        if re.search(r"transaction|orders|sales count|how many sales", ql):
            mname, agg, fmt = "Transactions", lambda g: len(g), num
        elif "unit" in ql:
            mname, agg, fmt = "Units sold", lambda g: g["Units_Sold"].sum(), num
        elif "discount" in ql:
            mname, agg, fmt = "Discount given", lambda g: g["Discount_Amount"].sum(), cr
        elif re.search(r"ticket|average|avg", ql):
            mname, agg, fmt = "Average ticket", lambda g: g["Revenue"].mean(), lakh
        else:
            mname, agg, fmt = "Net revenue", lambda g: g["Revenue"].sum(), cr

        dims = [("month", "Month", r"month"), ("Store_Location", "store", r"store|city|location"),
                ("Product_Category", "category", r"categor"), ("Region", "region", r"region|zone"),
                ("Brand", "brand", r"brand"), ("Payment_Mode", "payment mode", r"payment|pay mode"),
                ("Customer_Type", "customer type", r"customer|returning vs new"), ("Year", "year", r"\byear")]
        dim = next((c for c, _, rx in dims if re.search(rx, ql)), None)
        asc = bool(re.search(r"worst|lowest|least|bottom|smallest|weakest", ql))
        if dim is None:
            return dict(kind="scalar", label=mname, value=fmt(agg(d)), scope=scope, n=len(d), share=d["Revenue"].sum() / F["Revenue"].sum())
        key = d["Month"].dt.strftime("%b %Y") if dim == "month" else d[dim].astype(str)
        res = d.groupby(key).apply(agg).sort_values(ascending=asc)
        if dim == "month" and not re.search(r"worst|best|top|highest|lowest", ql):
            res = res.sort_index(key=lambda s: pd.to_datetime(s, format="%b %Y"))
        return dict(kind="rank", label=mname, dim=dim, res=res, fmt=fmt, asc=asc, scope=scope)

    if q.strip():
        out = answer(q)
        if out is None:
            st.info("I couldn't match that to the data. Try: “best store in 2024”, “revenue of Fashion in 2023”, “worst month by revenue”.")
        elif out["kind"] == "scalar":
            sc = " · ".join(out["scope"]) or "All data in current filters"
            html(f'<div class="answer"><div class="a-l">{out["label"]} · {sc}</div><div class="a-v">{out["value"]}</div>'
                 f'<div class="a-s">{out["n"]:,} transactions · {out["share"] * 100:.1f}% of filtered revenue</div></div>')
        else:
            res = out["res"]
            top = res.index[0]
            word = "Lowest" if out["asc"] else "Top"
            sc = " · ".join(out["scope"]) or "All data in current filters"
            show_n = res.head(10)
            html(f'<div class="answer"><div class="a-l">{out["label"]} by {out["dim"]} · {sc}</div>'
                 f'<div class="a-v">{word}: {top} — {out["fmt"](res.iloc[0])}</div>'
                 f'{bars_html(list(show_n.index), list(show_n.values), out["fmt"])}</div>')

    # ---- KPIs
    mrev = monthly()
    mtx = F.groupby("Month").size()
    mtk = F.groupby("Month")["Revenue"].mean()
    munit = monthly("Units_Sold")
    mdisc = monthly("Discount_Amount")
    mgross = monthly("Total_Sales")
    html('<div class="kpi-grid">'
         + kpi("Net revenue", cr(rev), yoy(rev, P["Revenue"].sum()), mrev.values, TEAL)
         + kpi("Gross sales", cr(gross), yoy(gross, P["Total_Sales"].sum()), mgross.values, BLUE)
         + kpi("Transactions", num(tx), yoy(tx, len(P)), mtx.values, VIOLET)
         + kpi("Avg ticket", lakh(ticket), yoy(ticket, P["Revenue"].mean() if len(P) else None), mtk.values, AMBER)
         + kpi("Units sold", num(units), yoy(units, P["Units_Sold"].sum()), munit.values, GREEN)
         + kpi("Discount given", cr(disc), yoy(disc, P["Discount_Amount"].sum()), mdisc.values, ROSE, good_up=False)
         + '</div>')

    # ---- the brief (computed insights)
    html('<div class="sec">The Meridian brief · generated from the filtered data</div>')
    cards = []
    # growth
    if has_prev and P["Revenue"].sum() > 0:
        g = yoy(rev, P["Revenue"].sum())
        tone = GREEN if g > 2 else (ROSE if g < -2 else AMBER)
        word = "grew" if g > 2 else ("fell" if g < -2 else "was essentially flat")
        cards.append(("Growth", f"Revenue {word} ({g:+.1f}%) vs the prior year",
                      f"{cr(rev)} against {cr(P['Revenue'].sum())} in the matching prior-year window.", tone))
    else:
        cards.append(("Growth", f"{cr(rev)} net revenue in scope",
                      "Pick a date range with a prior-year equivalent to see year-over-year change.", AMBER))
    # discount
    dp = disc / gross * 100 if gross else 0
    promo = F[F["Promotion_Applied"] == "Yes"]["Revenue"]
    nopromo = F[F["Promotion_Applied"] == "No"]["Revenue"]
    p_promo = stats.ttest_ind(promo, nopromo, equal_var=False).pvalue if len(promo) > 30 and len(nopromo) > 30 else np.nan
    sig = (not np.isnan(p_promo)) and p_promo < 0.05
    cards.append(("Discounts", f"{cr(disc)} given away ({dp:.1f}% of gross)",
                  f"Promotion vs non-promotion revenue per sale differs {'significantly' if sig else 'not significantly'} (p = {p_promo:.3f}). "
                  + ("Promotions appear to move ticket size." if sig else "No evidence promotions lift revenue per transaction."), ROSE if not sig else TEAL))
    # concentration
    cat_rev = F.groupby("Product_Category")["Revenue"].sum().sort_values(ascending=False)
    sto_rev = F.groupby("Store_Location")["Revenue"].sum().sort_values(ascending=False)
    gap = (sto_rev.iloc[0] / sto_rev.iloc[-1] - 1) * 100 if len(sto_rev) > 1 else 0
    cards.append(("Concentration", f"{cat_rev.index[0]} leads at {cat_rev.iloc[0] / rev * 100:.1f}% of revenue",
                  f"Top store {sto_rev.index[0]} is {gap:.1f}% ahead of the lowest ({sto_rev.index[-1]}). Categories and stores are "
                  + ("closely matched." if gap < 10 else "meaningfully spread."), BLUE))
    # seasonality
    daily = F.groupby("Date")["Revenue"].sum()
    p_seas = anova_p([g.values for _, g in daily.groupby(daily.index.month)])
    cv = mrev.std() / mrev.mean() * 100 if len(mrev) > 2 else np.nan
    cards.append(("Predictability", f"Monthly revenue varies only ±{cv:.1f}% around its mean" if not np.isnan(cv) else "Not enough months",
                  f"Seasonality across calendar months is {'statistically present' if p_seas < 0.05 else 'not statistically significant'} "
                  f"(p = {p_seas:.3f}). " + ("Plan around the seasonal pattern." if p_seas < 0.05 else "Treat demand as a stable level plus noise."), AMBER))
    # customers
    ret = (F["Customer_Type"] == "Returning").mean() * 100
    tk = F.groupby("Customer_Type")["Revenue"].mean()
    diff = (tk.get("Returning", np.nan) / tk.get("New", np.nan) - 1) * 100 if {"Returning", "New"} <= set(tk.index) else np.nan
    cards.append(("Customers", f"{ret:.1f}% of transactions are returning customers",
                  f"Returning customers' average ticket is {diff:+.1f}% vs new customers." if not np.isnan(diff) else "Customer type not available.", VIOLET))
    # best / worst month
    if len(mrev) > 1:
        bm, wm = mrev.idxmax(), mrev.idxmin()
        cards.append(("Peaks & troughs", f"Best month {bm:%b %Y}, weakest {wm:%b %Y}",
                      f"{cr(mrev.max())} vs {cr(mrev.min())}, a {(mrev.max() / mrev.min() - 1) * 100:.1f}% swing.", GREEN))
    html('<div class="brief-grid">' + "".join(
        f'<div class="brief" style="--c:{c}"><div class="brief-t">{t}</div><div class="brief-h">{h}</div><div class="brief-b">{b}</div></div>'
        for t, h, b, c in cards) + '</div>')

    c1, c2 = st.columns([1.7, 1])
    with c1, st.container(border=True):
        card("Monthly net revenue", "Trend with 3-month moving average")
        mm = mrev.reset_index()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=mm["Month"], y=mm["Revenue"], name="Revenue", mode="lines+markers",
                                 line=dict(color=TEAL, width=2.5), fill="tozeroy", fillcolor="rgba(45,212,191,.08)"))
        fig.add_trace(go.Scatter(x=mm["Month"], y=mm["Revenue"].rolling(3).mean(), name="3-mo avg",
                                 line=dict(color=AMBER, width=2, dash="dot")))
        style(fig, 330, legend=True).update_yaxes(tickformat=",.3s", rangemode="tozero")
        show(fig)
    with c2, st.container(border=True):
        card("Revenue by store", "Ranked, coloured by region")
        s = F.groupby(["Store_Location", "Region"], as_index=False)["Revenue"].sum().sort_values("Revenue")
        fig = px.bar(s, x="Revenue", y="Store_Location", color="Region", orientation="h", color_discrete_sequence=SEQ)
        style(fig, 330, legend=True).update_xaxes(tickformat=",.3s")
        show(fig)

# ═════════════════════════ PAGE: PULSE ═════════════════════════
elif page == "Pulse":
    html('<div class="eyebrow">01 · Pulse</div><div class="page-h">How the business <em>breathes</em></div>'
         '<div class="page-p">Daily, weekly and monthly rhythm for the current filters.</div>')
    m1, m2 = st.columns([1, 1])
    gran = m1.radio("Granularity", ["Daily", "Weekly", "Monthly"], index=2, horizontal=True)
    metric = m2.radio("Metric", ["Revenue", "Transactions", "Units"], horizontal=True)
    col = {"Revenue": "Revenue", "Transactions": "Revenue", "Units": "Units_Sold"}[metric]
    agg = "count" if metric == "Transactions" else "sum"
    rule = {"Daily": "D", "Weekly": "W", "Monthly": "MS"}[gran]
    ts = F.set_index("Date")[col].resample(rule).agg(agg)
    if gran != "Daily" and len(ts) > 2:  # drop partial edge periods only for weekly/monthly
        ts = ts.iloc[1:-1] if gran == "Weekly" else ts
    with st.container(border=True):
        card(f"{gran} {metric.lower()}", "With moving average" if gran == "Daily" else None)
        fig = go.Figure(go.Scatter(x=ts.index, y=ts.values, mode="lines", name=metric, line=dict(color=TEAL, width=1.8),
                                   fill="tozeroy", fillcolor="rgba(45,212,191,.10)"))
        if gran == "Daily":
            fig.add_trace(go.Scatter(x=ts.index, y=ts.rolling(30, min_periods=7).mean(), name="30-day avg", line=dict(color=AMBER, width=2.4)))
        style(fig, 340, legend=gran == "Daily").update_yaxes(tickformat=",.3s", rangemode="tozero")
        show(fig)

    c1, c2 = st.columns(2)
    with c1, st.container(border=True):
        card("Year over year · monthly revenue", "Each line is a calendar year")
        mm = F.groupby(["Year", F["Date"].dt.month])["Revenue"].sum().reset_index(name="Revenue")
        mm.columns = ["Year", "Mo", "Revenue"]
        mm["Mon"] = pd.to_datetime(mm["Mo"], format="%m").dt.strftime("%b")
        fig = px.line(mm, x="Mon", y="Revenue", color=mm["Year"].astype(str), markers=True, color_discrete_sequence=[BLUE, AMBER, TEAL])
        style(fig, 320, legend=True).update_yaxes(tickformat=",.3s")
        fig.update_xaxes(categoryorder="array", categoryarray=list(pd.to_datetime(range(1, 13), format="%m").strftime("%b")))
        show(fig)
        ys = sorted(F["Year"].unique())
        if len(ys) >= 2:
            a, b = F[F["Year"] == ys[-2]]["Revenue"].sum(), F[F["Year"] == ys[-1]]["Revenue"].sum()
            st.caption(f"{ys[-1]} vs {ys[-2]} (selected range): {pct((b / a - 1) * 100)}")
    with c2, st.container(border=True):
        card("Seasonality check · deviation from average", "Average daily revenue by calendar month")
        dly = F.groupby("Date")["Revenue"].sum()
        sm = dly.groupby(dly.index.month).mean()
        dev = (sm / sm.mean() - 1) * 100
        lab = pd.to_datetime(dev.index, format="%m").strftime("%b")
        fig = go.Figure(go.Bar(x=lab, y=dev.values, marker_color=[GREEN if v >= 0 else ROSE for v in dev.values],
                               hovertemplate="%{x}: %{y:+.1f}%<extra></extra>"))
        style(fig, 320).update_yaxes(ticksuffix="%")
        show(fig)
        p = anova_p([g.values for _, g in dly.groupby(dly.index.month)])
        st.caption(f"One-way ANOVA across months: p = {p:.3f} → " + ("seasonality is statistically significant." if p < 0.05 else "no reliable seasonal effect."))

    c3, c4 = st.columns(2)
    with c3, st.container(border=True):
        card("Day-of-week profile", "Average revenue per day")
        dw = dly.groupby(dly.index.dayofweek).mean()
        names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        fig = go.Figure(go.Bar(x=[names[i] for i in dw.index], y=dw.values, marker_color=VIOLET))
        style(fig, 280).update_yaxes(tickformat=",.3s", rangemode="tozero")
        show(fig)
    with c4, st.container(border=True):
        card("Holiday effect", "Average daily revenue, holiday vs regular day")
        hd = F.groupby("Date").agg(R=("Revenue", "sum"), H=("Holiday_Flag", "max"))
        h = hd.groupby("H")["R"].mean()
        if len(h) == 2:
            lift = (h.get(1, np.nan) / h.get(0, np.nan) - 1) * 100
            p_h = stats.ttest_ind(hd[hd.H == 1]["R"], hd[hd.H == 0]["R"], equal_var=False).pvalue
            fig = go.Figure(go.Bar(x=["Regular day", "Holiday"], y=[h.get(0), h.get(1)], marker_color=[BLUE, AMBER]))
            style(fig, 230).update_yaxes(tickformat=",.3s", rangemode="tozero")
            show(fig)
            st.caption(f"Holiday lift {pct(lift)} · Welch t-test p = {p_h:.3f}")
        else:
            st.info("No holiday flag variation in the selected range.")

    with st.container(border=True):
        years_avail = sorted(F["Year"].unique())
        cy = st.radio("Calendar year", years_avail, index=len(years_avail) - 1, horizontal=True)
        card("Calendar heatmap", "Each square is a day. Brighter = above typical revenue")
        dd = dly[dly.index.year == cy]
        if not dd.empty:
            jan1 = pd.Timestamp(year=cy, month=1, day=1)
            wk = ((dd.index - jan1).days + jan1.dayofweek) // 7
            z = np.full((7, wk.max() + 1), np.nan)
            txt = np.full((7, wk.max() + 1), "", dtype=object)
            for dt, w, v in zip(dd.index, wk, dd.values):
                z[dt.dayofweek, w] = v
                txt[dt.dayofweek, w] = f"{dt:%d %b %Y}: {cr(v)}"
            fig = go.Figure(go.Heatmap(z=z, text=txt, hoverinfo="text", xgap=3, ygap=3,
                                       colorscale=[[0, "#13203A"], [.5, "#0F766E"], [1, "#2DD4BF"]], showscale=False,
                                       zmin=np.nanpercentile(z, 5), zmax=np.nanpercentile(z, 95)))
            tick_w = [((pd.Timestamp(year=cy, month=m, day=1) - jan1).days + jan1.dayofweek) // 7 for m in range(1, 13)]
            style(fig, 230).update_xaxes(tickmode="array", tickvals=tick_w, ticktext=list(pd.to_datetime(range(1, 13), format="%m").strftime("%b")), showgrid=False)
            fig.update_yaxes(tickmode="array", tickvals=list(range(7)), ticktext=names, autorange="reversed", showgrid=False)
            show(fig)

# ═════════════════════════ PAGE: MIX ═════════════════════════
elif page == "Mix":
    html('<div class="eyebrow">02 · Mix</div><div class="page-h">What sells, <em>and to whom</em></div>'
         '<div class="page-p">Category, brand, customer and payment composition.</div>')
    c1, c2 = st.columns([1, 1.5])
    with c1, st.container(border=True):
        card("Category mix", "Share of net revenue")
        s = F.groupby("Product_Category")["Revenue"].sum().reset_index()
        fig = go.Figure(go.Pie(labels=s["Product_Category"], values=s["Revenue"], hole=.66, sort=True,
                               marker=dict(colors=SEQ, line=dict(color="#0C1325", width=3)), textinfo="percent"))
        fig.add_annotation(text=f"<b>{cr(rev)}</b>", showarrow=False, font=dict(size=15, color="#fff"))
        style(fig, 340, legend=True)
        fig.update_layout(legend=dict(orientation="h", y=-0.08))
        show(fig)
    with c2, st.container(border=True):
        card("Monthly revenue by category", "Stacked")
        s = F.groupby(["Month", "Product_Category"], as_index=False)["Revenue"].sum()
        fig = px.bar(s, x="Month", y="Revenue", color="Product_Category", color_discrete_sequence=SEQ)
        style(fig, 340, legend=True).update_yaxes(tickformat=",.3s")
        show(fig)

    c3, c4 = st.columns([1.3, 1])
    with c3, st.container(border=True):
        card("Store × category index", "100 = that store's average. Green over-indexes, purple under-indexes")
        pv = F.pivot_table(index="Store_Location", columns="Product_Category", values="Revenue", aggfunc="sum")
        idx = pv.div(pv.mean(axis=1), axis=0) * 100
        fig = go.Figure(go.Heatmap(z=idx.values, x=idx.columns, y=idx.index, colorscale="Viridis", zmid=100,
                                   text=np.round(idx.values, 0), texttemplate="%{text:.0f}", xgap=2, ygap=2,
                                   colorbar=dict(thickness=10, len=.8)))
        style(fig, 340)
        show(fig)
    with c4, st.container(border=True):
        card("Top brands", "Net revenue")
        b = F.groupby("Brand")["Revenue"].sum().nlargest(10).sort_values()
        fig = go.Figure(go.Bar(x=b.values, y=b.index, orientation="h", marker_color=TEAL))
        style(fig, 340).update_xaxes(tickformat=",.3s")
        show(fig)

    with st.container(border=True):
        card("Category → subcategory → brand", "Click a block to drill in")
        t = F.groupby(["Product_Category", "Product_Subcategory", "Brand"], as_index=False)["Revenue"].sum()
        fig = px.treemap(t, path=[px.Constant("All"), "Product_Category", "Product_Subcategory", "Brand"], values="Revenue",
                         color="Product_Category", color_discrete_sequence=SEQ)
        fig.update_traces(marker=dict(line=dict(color="#0C1325", width=2)), textfont=dict(color="#fff"))
        style(fig, 420)
        show(fig)

    c5, c6 = st.columns(2)
    with c5, st.container(border=True):
        card("Payment mode", "Share of revenue")
        s = F.groupby("Payment_Mode")["Revenue"].sum().reset_index()
        fig = go.Figure(go.Pie(labels=s["Payment_Mode"], values=s["Revenue"], hole=.6, marker=dict(colors=SEQ, line=dict(color="#0C1325", width=3))))
        style(fig, 280, legend=True)
        show(fig)
    with c6, st.container(border=True):
        card("New vs returning customers", "Revenue and average ticket")
        s = F.groupby("Customer_Type").agg(Revenue=("Revenue", "sum"), Ticket=("Revenue", "mean"), Tx=("Revenue", "size")).reset_index()
        fig = go.Figure(go.Pie(labels=s["Customer_Type"], values=s["Revenue"], hole=.6, marker=dict(colors=[AMBER, VIOLET], line=dict(color="#0C1325", width=3))))
        style(fig, 220, legend=True)
        show(fig)
        s["Revenue"] = s["Revenue"].map(cr); s["Ticket"] = s["Ticket"].map(lakh)
        table(s.rename(columns={"Customer_Type": "Customer", "Tx": "Transactions"}))

# ═════════════════════════ PAGE: STORES ═════════════════════════
elif page == "Stores":
    html('<div class="eyebrow">03 · Stores</div><div class="page-h">Eight stores, <em>side by side</em></div>'
         '<div class="page-p">Leaderboard, quality and trajectory by location.</div>')
    S = F.groupby(["Store_ID", "Store_Location", "Region"]).agg(
        Revenue=("Revenue", "sum"), Tx=("Revenue", "size"), Units=("Units_Sold", "sum"),
        Ticket=("Revenue", "mean"), Rating=("Store_Rating", "mean"), Disc=("Discount_Amount", "sum")).reset_index().sort_values("Revenue", ascending=False)
    S["Share"] = S["Revenue"] / S["Revenue"].sum() * 100
    if has_prev:
        pr = P.groupby("Store_Location")["Revenue"].sum()
        S["YoY"] = S["Store_Location"].map(lambda s_: (S.loc[S.Store_Location == s_, "Revenue"].iloc[0] / pr.get(s_, np.nan) - 1) * 100)
    else:
        S["YoY"] = np.nan

    c1, c2 = st.columns([1, 1.15])
    with c1, st.container(border=True):
        card("Leaderboard", "Net revenue")
        html('<div style="margin-top:6px">' + bars_html([f"{a} · {b}" for a, b in zip(S["Store_ID"], S["Store_Location"])], list(S["Revenue"]), cr) + "</div>")
        p_st = anova_p([g["Revenue"].values for _, g in F.groupby("Store_Location")])
        gap = (S["Revenue"].iloc[0] / S["Revenue"].iloc[-1] - 1) * 100
        st.caption(f"Top vs bottom store: {gap:.1f}% apart · transaction value differs across stores: " + ("significant" if p_st < 0.05 else "not significant") + f" (p = {p_st:.3f}).")
    with c2, st.container(border=True):
        card("Store trajectories", "Monthly revenue by store")
        s = F.groupby(["Month", "Store_Location"], as_index=False)["Revenue"].sum()
        fig = px.line(s, x="Month", y="Revenue", color="Store_Location", color_discrete_sequence=SEQ)
        style(fig, 330, legend=True).update_yaxes(tickformat=",.3s")
        show(fig)

    c3, c4 = st.columns(2)
    with c3, st.container(border=True):
        card("Rating vs revenue", "Bubble = transactions")
        fig = px.scatter(S, x="Rating", y="Revenue", size="Tx", color="Region", hover_name="Store_Location", size_max=38, color_discrete_sequence=SEQ)
        style(fig, 320, legend=True).update_yaxes(tickformat=",.3s")
        show(fig)
    with c4, st.container(border=True):
        card("Average ticket by store", "Net revenue per transaction")
        s2 = S.sort_values("Ticket")
        fig = go.Figure(go.Bar(x=s2["Ticket"], y=s2["Store_Location"], orientation="h", marker_color=AMBER,
                               text=[lakh(v) for v in s2["Ticket"]], textposition="inside"))
        style(fig, 320).update_xaxes(tickformat=",.3s")
        show(fig)

    card("Store scorecard")
    out = S.copy()
    out["Revenue (₹ Cr)"] = out["Revenue"] / 1e7
    out["Avg ticket (₹ L)"] = out["Ticket"] / 1e5
    out["Discount (₹ Cr)"] = out["Disc"] / 1e7
    table(out[["Store_ID", "Store_Location", "Region", "Revenue (₹ Cr)", "Share", "YoY", "Tx", "Units", "Avg ticket (₹ L)", "Rating", "Discount (₹ Cr)"]].rename(
        columns={"Store_ID": "ID", "Store_Location": "Store", "Share": "Share %", "YoY": "YoY %", "Tx": "Transactions"}),
        column_config={"Revenue (₹ Cr)": st.column_config.ProgressColumn(format="%.1f", min_value=0, max_value=float(out["Revenue (₹ Cr)"].max())),
                       "Share %": st.column_config.NumberColumn(format="%.1f%%"), "YoY %": st.column_config.NumberColumn(format="%+.1f%%"),
                       "Rating": st.column_config.NumberColumn(format="%.2f ⭐"), "Avg ticket (₹ L)": st.column_config.NumberColumn(format="%.2f"),
                       "Discount (₹ Cr)": st.column_config.NumberColumn(format="%.1f")})

# ═════════════════════════ PAGE: FORECAST ═════════════════════════
elif page == "Forecast":
    html('<div class="eyebrow">04 · Forecast</div><div class="page-h">One <em>honest</em> forecast</div>'
         '<div class="page-p">Four models are back-tested on the most recent days; the winner is refit on all history. '
         'If nothing beats a flat average, we say so.</div>')

    @st.cache_data(show_spinner="Back-testing forecasting models…")
    def run_forecast(series: pd.Series, horizon: int):
        s = series.asfreq("D").fillna(0.0).astype(float)
        n_test = 90 if len(s) >= 270 else max(14, len(s) // 4)
        tr, te = s.iloc[:-n_test], s.iloc[-n_test:]

        def feats(idx, t0):
            X = np.zeros((len(idx), 7 + 12 + 1))
            X[np.arange(len(idx)), idx.dayofweek] = 1
            X[np.arange(len(idx)), 7 + idx.month - 1] = 1
            X[:, -1] = (np.arange(len(idx)) + t0) / 365.0
            return X

        def pred(name, train, idx):
            if name == "Flat average (365d)":
                return np.repeat(train.iloc[-365:].mean(), len(idx))
            if name == "Weekly seasonal naive":
                prof = train.iloc[-56:].groupby(train.iloc[-56:].index.dayofweek).mean()
                return np.array([prof.get(d, train.mean()) for d in idx.dayofweek])
            if name == "Ridge (calendar + trend)":
                mu = train.mean()
                m = Ridge(alpha=3.0).fit(feats(train.index, 0), train.values / mu)
                return m.predict(feats(idx, len(train))) * mu
            if name == "Holt-Winters (weekly)":
                f = ExponentialSmoothing(train.iloc[-365:], trend=None, seasonal="add", seasonal_periods=7,
                                         initialization_method="estimated").fit()
                return np.asarray(f.forecast(len(idx)))
            raise ValueError(name)

        names = ["Flat average (365d)", "Weekly seasonal naive", "Ridge (calendar + trend)", "Holt-Winters (weekly)"]
        rows, preds = [], {}
        for nme in names:
            try:
                p = pred(nme, tr, te.index)
                preds[nme] = p
                wape = np.abs(te.values - p).sum() / np.abs(te.values).sum() * 100
                rows.append({"Model": nme, "WAPE %": wape, "MAE (₹ L/day)": np.abs(te.values - p).mean() / 1e5})
            except Exception:
                continue
        res = pd.DataFrame(rows).sort_values("WAPE %").reset_index(drop=True)
        best = res.iloc[0]["Model"]
        resid = te.values - preds[best]
        fidx = pd.date_range(s.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")
        fc = np.maximum(pred(best, s, fidx), 0)
        lo, hi = np.quantile(resid, [0.10, 0.90])
        return dict(res=res, best=best, fidx=fidx, fc=fc, lo=np.maximum(fc + lo, 0), hi=fc + hi, n_test=n_test, hist=s,
                    test_idx=te.index, test_pred=preds[best], test=te)

    daily_f = F.groupby("Date")["Revenue"].sum()
    if len(daily_f) < 120:
        st.warning("Select at least ~120 days of data to run forecasting.")
    else:
        hz = st.slider("Forecast horizon (days)", 30, 180, 90, 15)
        R = run_forecast(daily_f, hz)
        res = R["res"]
        flat = res[res["Model"] == "Flat average (365d)"]["WAPE %"].iloc[0]
        gain = flat - res.iloc[0]["WAPE %"]
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Selected model", R["best"].split(" (")[0])
        k2.metric(f"Expected revenue · next {hz}d", cr(R["fc"].sum()))
        k3.metric("Run-rate", cr(R["fc"].mean() * 30) + " / 30d")
        k4.metric("Back-test error (WAPE)", f"{res.iloc[0]['WAPE %']:.1f}%")

        if gain < 1.0:
            st.info(f"**Honest read:** no model beats a flat average by a meaningful margin (gain {gain:.2f} pts). Daily revenue behaves like a stable level plus noise, "
                    f"so use this as a planning range, not a precise prediction.")
        else:
            st.success(f"**{R['best']}** beats the flat-average baseline by {gain:.1f} WAPE points on the last {R['n_test']} days.")

        with st.container(border=True):
            card(f"Daily revenue forecast · {hz} days", "Shaded band = 80% interval from back-test errors")
            h = R["hist"].iloc[-240:]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=h.index, y=h.values, name="Actual", line=dict(color=BLUE, width=1.4)))
            fig.add_trace(go.Scatter(x=list(R["fidx"]) + list(R["fidx"][::-1]), y=list(R["hi"]) + list(R["lo"][::-1]), fill="toself",
                                     fillcolor="rgba(245,165,36,.14)", line=dict(width=0), name="80% range", hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=R["fidx"], y=R["fc"], name="Forecast", line=dict(color=AMBER, width=2.4)))
            fig.add_vline(x=h.index[-1], line=dict(color="#586784", dash="dot"))
            style(fig, 380, legend=True).update_yaxes(tickformat=",.3s", rangemode="tozero")
            show(fig)
        c1, c2 = st.columns([1, 1])
        with c1, st.container(border=True):
            card("Model leaderboard", f"Back-test on the last {R['n_test']} days · lower is better")
            t = res.copy()
            t["WAPE %"] = t["WAPE %"].round(2); t["MAE (₹ L/day)"] = t["MAE (₹ L/day)"].round(2)
            table(t)
        with c2, st.container(border=True):
            card("Back-test: actual vs predicted", f"Winner on the held-out {R['n_test']} days")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=R["test_idx"], y=R["test"].values, name="Actual", line=dict(color=BLUE, width=1.4)))
            fig.add_trace(go.Scatter(x=R["test_idx"], y=R["test_pred"], name="Predicted", line=dict(color=AMBER, width=2)))
            style(fig, 260, legend=True).update_yaxes(tickformat=",.3s")
            show(fig)
        ft = pd.DataFrame({"Date": R["fidx"], "Forecast (₹ L)": R["fc"] / 1e5, "Low (₹ L)": R["lo"] / 1e5, "High (₹ L)": R["hi"] / 1e5})
        mf = ft.set_index("Date").resample("MS").sum().reset_index()
        card("Monthly roll-up of the forecast")
        mf["Month"] = mf["Date"].dt.strftime("%b %Y")
        table(mf[["Month", "Forecast (₹ L)", "Low (₹ L)", "High (₹ L)"]].round(1))
        st.download_button("⬇ Download daily forecast (CSV)", ft.to_csv(index=False).encode("utf-8"), "revenue_forecast.csv", "text/csv")

# ═════════════════════════ PAGE: DISCOUNT LAB ═════════════════════════
elif page == "Discount Lab":
    html('<div class="eyebrow">05 · Discount Lab</div><div class="page-h">Do discounts <em>actually</em> pay?</div>'
         '<div class="page-p">Statistical tests on whether promotions and discount depth change revenue or volume.</div>')
    promo_share = (F["Promotion_Applied"] == "Yes").mean() * 100
    k = st.columns(4)
    k[0].metric("Discount given", cr(disc))
    k[1].metric("% of gross sales", f"{disc / gross * 100:.2f}%")
    k[2].metric("Avg discount depth", f"{F['Discount_Percentage'].mean():.1f}%")
    k[3].metric("Transactions with promo", f"{promo_share:.1f}%")

    a = F[F["Promotion_Applied"] == "Yes"]["Revenue"]; b_ = F[F["Promotion_Applied"] == "No"]["Revenue"]
    t_p = stats.ttest_ind(a, b_, equal_var=False).pvalue if len(a) > 30 and len(b_) > 30 else np.nan
    c1, c2 = st.columns([1, 1.4])
    with c1, st.container(border=True):
        card("Do promotions work?", "Average revenue per transaction")
        fig = go.Figure(go.Bar(x=["With promo", "No promo"], y=[a.mean(), b_.mean()], marker_color=[AMBER, BLUE],
                               text=[lakh(a.mean()), lakh(b_.mean())], textposition="inside"))
        style(fig, 280).update_yaxes(tickformat=",.3s", rangemode="tozero")
        show(fig)
        verdict = "significant difference" if t_p < 0.05 else "not significant"
        st.markdown(f"**Welch t-test p = {t_p:.4f}** → {verdict}")
        st.caption("A p-value above 0.05 means the observed gap is within random variation.")
    with c2, st.container(border=True):
        card("Discount depth vs volume and ticket", "By discount percentage band")
        g = F.groupby("Discount_Percentage").agg(Units=("Units_Sold", "mean"), Ticket=("Revenue", "mean"), Tx=("Revenue", "size")).reset_index()
        fig = go.Figure()
        fig.add_trace(go.Bar(x=g["Discount_Percentage"].astype(str) + "%", y=g["Units"], name="Avg units / sale", marker_color=VIOLET, yaxis="y"))
        fig.add_trace(go.Scatter(x=g["Discount_Percentage"].astype(str) + "%", y=g["Ticket"], name="Avg ticket", mode="lines+markers",
                                 line=dict(color=AMBER, width=2.5), yaxis="y2"))
        style(fig, 300, legend=True)
        fig.update_layout(yaxis2=dict(overlaying="y", side="right", tickformat=",.3s", showgrid=False, tickfont=dict(color=MUTED)))
        show(fig)
        r, p_r = stats.pearsonr(F["Discount_Percentage"], F["Units_Sold"])
        st.caption(f"Correlation of discount % with units per sale: r = {r:+.4f} (p = {p_r:.3f}). "
                   + ("Deeper discounts do not reliably sell more units." if p_r >= 0.05 else "There is a statistically detectable relationship."))

    c3, c4 = st.columns(2)
    with c3, st.container(border=True):
        card("Where the discount money goes", "₹ Cr given away by category")
        s = F.groupby("Product_Category")["Discount_Amount"].sum().sort_values()
        fig = go.Figure(go.Bar(x=s.values, y=s.index, orientation="h", marker_color=ROSE))
        style(fig, 280).update_xaxes(tickformat=",.3s")
        show(fig)
    with c4, st.container(border=True):
        card("What-if simulator", "Cut average discount depth and see the revenue effect")
        cut = st.slider("Reduce every discount by (percentage points)", 0, 15, 5)
        slope, _, _, p_s, _ = stats.linregress(F["Discount_Percentage"], F["Units_Sold"])
        eff_slope = slope if p_s < 0.05 else 0.0
        new_pct = (F["Discount_Percentage"] - cut).clip(lower=0)
        vol = 1 + (eff_slope * (new_pct - F["Discount_Percentage"])) / F["Units_Sold"].mean()
        new_rev = (F["Total_Sales"] * vol * (1 - new_pct / 100)).sum()
        delta = new_rev - rev
        st.metric("Projected revenue change", cr(delta), f"{delta / rev * 100:+.2f}%")
        st.caption("Scenario model: volume response is taken from the measured slope of units vs discount "
                   + ("(statistically significant)." if p_s < 0.05 else "(not significant, so volume is held constant).") + " Treat as directional.")

# ═════════════════════════ PAGE: SEGMENTS ═════════════════════════
elif page == "Segments":
    html('<div class="eyebrow">06 · Segments</div><div class="page-h">Which stores <em>behave alike?</em></div>'
         '<div class="page-p">K-Means clustering on standardised performance features, projected to 2-D with PCA.</div>')
    level = st.radio("Level", ["Store", "Store × category"], horizontal=True)
    keys = ["Store_Location"] if level == "Store" else ["Store_Location", "Product_Category"]
    g = F.groupby(keys)
    X_ = pd.DataFrame({
        "Revenue (₹ Cr)": g["Revenue"].sum() / 1e7,
        "Avg ticket (₹ L)": g["Revenue"].mean() / 1e5,
        "Units / sale": g["Units_Sold"].mean(),
        "Avg discount %": g["Discount_Percentage"].mean(),
        "Promo share %": g["Promotion_Applied"].apply(lambda x: (x == "Yes").mean() * 100),
        "Returning %": g["Customer_Type"].apply(lambda x: (x == "Returning").mean() * 100),
        "Rating": g["Store_Rating"].mean(),
    })
    yrs = sorted(F["Year"].unique())
    if len(yrs) >= 2:
        pv = F.pivot_table(index=keys, columns="Year", values="Revenue", aggfunc="sum")
        X_["YoY growth %"] = (pv[yrs[-1]] / pv[yrs[-2]].replace(0, np.nan) - 1) * 100
    X_ = X_.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="all").dropna()
    if len(X_) < 4:
        st.info("Need at least 4 stores/groups in the filter to segment. Widen the filters.")
    else:
        Z = StandardScaler().fit_transform(X_)
        kmax = min(6, len(X_) - 1)
        k = st.slider("Number of segments", 2, kmax, min(4, kmax))
        km = KMeans(n_clusters=k, n_init=20, random_state=42).fit(Z)
        sil = silhouette_score(Z, km.labels_)
        zdf = pd.DataFrame(Z, columns=X_.columns, index=X_.index)
        zdf["c"] = km.labels_
        names_, used = {}, set()
        for c in range(k):
            m = zdf[zdf.c == c].drop(columns="c").mean()
            f_ = m.abs().idxmax()
            nm = f"{'High' if m[f_] > 0 else 'Low'} {f_.split(' (')[0].lower()}"
            if nm in used:
                nm += f" #{c + 1}"
            used.add(nm); names_[c] = nm
        X_["Segment"] = [names_[c] for c in km.labels_]
        pc = PCA(2, random_state=42).fit(Z)
        P2 = pc.transform(Z)
        X_["pc1"], X_["pc2"] = P2[:, 0], P2[:, 1]
        lab = X_.reset_index()
        lab["Label"] = lab[keys].astype(str).agg(" · ".join, axis=1)
        m1, m2, m3 = st.columns(3)
        m1.metric("Silhouette score", f"{sil:.3f}")
        m2.metric("Variance explained (2 PCs)", f"{pc.explained_variance_ratio_.sum() * 100:.0f}%")
        m3.metric("Groups clustered", len(X_))
        st.caption("Silhouette near 0 means clusters overlap; ≥0.5 is strong separation. " + ("Segments here are weakly separated, so read them as tendencies." if sil < 0.25 else "Segments are reasonably distinct."))
        with st.container(border=True):
            card("Segment map (PCA view)", "Hover for the store")
            fig = px.scatter(lab, x="pc1", y="pc2", color="Segment", hover_name="Label", size="Revenue (₹ Cr)", size_max=30, color_discrete_sequence=SEQ)
            fig.update_traces(marker=dict(line=dict(color="#0C1325", width=1.5)))
            style(fig, 400, legend=True)
            fig.update_xaxes(title="PC1"); fig.update_yaxes(title="PC2")
            show(fig)
        with st.container(border=True):
            card("Segment profile", "Average of each feature within the segment")
            prof = X_.drop(columns=["pc1", "pc2"]).groupby("Segment").mean().round(2)
            prof.insert(0, "Members", X_.groupby("Segment").size())
            table(prof.reset_index())
        table(lab.drop(columns=["pc1", "pc2", "Label"]).round(2))
        st.download_button("⬇ Download segments (CSV)", lab.drop(columns=["pc1", "pc2", "Label"]).to_csv(index=False).encode("utf-8"), "store_segments.csv", "text/csv")

# ═════════════════════════ PAGE: DATA & METHODS ═════════════════════════
else:
    html('<div class="eyebrow">07 · Integrity & Methods</div><div class="page-h">Can you <em>trust</em> these numbers?</div>'
         '<div class="page-p">Automated data-quality checks on the full dataset (filters do not apply here).</div>')
    chk = []
    chk.append(("Rows / columns", f"{len(df):,} × {df.shape[1]}", "ok"))
    chk.append(("Date coverage", f"{df.Date.min():%d %b %Y} → {df.Date.max():%d %b %Y} · {df.Date.nunique()} of {(df.Date.max() - df.Date.min()).days + 1} days", "ok" if df.Date.nunique() == (df.Date.max() - df.Date.min()).days + 1 else "warn"))
    chk.append(("Missing values", f"{int(df.isna().sum().sum())} cells", "ok" if df.isna().sum().sum() == 0 else "warn"))
    chk.append(("Duplicate rows", f"{int(df.duplicated().sum())}", "ok" if df.duplicated().sum() == 0 else "warn"))
    chk.append(("Negative / zero revenue", f"{int((df.Revenue <= 0).sum())} rows", "ok" if (df.Revenue <= 0).sum() == 0 else "warn"))
    if "Unit_Price" in df.columns:
        r1 = (np.abs(df.Total_Sales - df.Unit_Price * df.Units_Sold) < 1).mean() * 100
        chk.append(("Total_Sales = Unit_Price × Units", f"{r1:.1f}% of rows reconcile", "ok" if r1 > 99 else "warn"))
    r2 = (np.abs(df.Revenue - df.Total_Sales * (1 - df.Discount_Percentage / 100)) < 1).mean() * 100
    chk.append(("Revenue = Total_Sales × (1 − discount)", f"{r2:.1f}% of rows reconcile", "ok" if r2 > 99 else "warn"))
    mp = df.groupby("Store_ID")["Store_Location"].nunique().max()
    chk.append(("Store ID ↔ location mapping", "1-to-1" if mp == 1 else "inconsistent", "ok" if mp == 1 else "warn"))
    table(pd.DataFrame(chk, columns=["Check", "Result", "Status"]).assign(Status=lambda d: d.Status.map({"ok": "✅ Pass", "warn": "⚠️ Review"})))

    c1, c2 = st.columns(2)
    with c1, st.container(border=True):
        card("Methodology")
        st.markdown("""
- **Net revenue** = `Revenue` column (after discount). **Gross sales** = `Total_Sales`. **Discount** = gross − net.
- **Year-over-year** compares the selected date window with the same window one year earlier, under the same store/region/category filters.
- **Significance tests:** Welch t-test (promotion effect, holiday effect), one-way ANOVA (seasonality, store differences), Pearson r (discount vs units).
- **Forecast:** daily revenue; four models back-tested on the latest 90 days; metric is WAPE; interval is the 10th–90th percentile of back-test errors.
- **Segments:** K-Means on standardised features, 2-D view via PCA, quality via silhouette score.
""")
    with c2, st.container(border=True):
        card("Column dictionary")
        table(pd.DataFrame({"Column": df.columns, "Type": [str(t) for t in df.dtypes], "Unique": [df[c].nunique() for c in df.columns]}))
    with st.expander("Browse the filtered transactions"):
        table(F.head(1000))

html('<div class="foot">Meridian · Retail Intelligence Observatory — built with Python, Streamlit, Plotly, scikit-learn and statsmodels. '
     'Forecasts and clusters are decision-support outputs and should be validated before operational use.</div>')
