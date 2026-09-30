import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from common import STATUS_ICON, as_of, load, md, money, palette, signed_money, style_fig

AS_OF = as_of()
p = palette()

pnl = load("fct_monthly_pnl")
bud = load("fct_budget_monthly")
live = load("forecast_live")
outlook = load("fy_outlook")
flags = load("variance_monthly")
acc = load("accuracy_summary").iloc[0]


def group(df: pd.DataFrame) -> pd.Series:
    return pd.Series(np.where(df["account_type"] == "Revenue", "Revenue", "Costs"), index=df.index)


pnl["group"], bud["group"], live["group"], outlook["group"] = group(pnl), group(bud), group(live), group(outlook)
pnl["ds"] = pd.PeriodIndex(pnl["period"], freq="M").to_timestamp()
bud["ds"] = pd.PeriodIndex(bud["period"], freq="M").to_timestamp()
act_m = pnl.pivot_table(index="ds", columns="group", values="actual_usd", aggfunc="sum")
bud_m = bud.pivot_table(index="ds", columns="group", values="budget_usd", aggfunc="sum")
fc_m = live.pivot_table(index="ds", columns="group", values="yhat", aggfunc="sum")
for t in (act_m, bud_m, fc_m):
    t["Operating income"] = t["Revenue"] - t["Costs"]

st.title("FP&A rolling forecast & variance radar")
st.markdown(
    f"**{AS_OF.strftime('%B %Y')} close.** Every month the system rebuilds the P&L from the general ledger, "
    "re-forecasts the next 12 months with whichever method has been most accurate for each line, "
    "and flags the lines that are both **material** and **outside the expected range**, with a first-draft explanation."
)

# --- this month -----------------------------------------------------------------------------
now = AS_OF.to_timestamp()
st.subheader(f"{AS_OF.strftime('%B %Y')}: actual and variance vs budget")
c1, c2, c3, c4 = st.columns(4)
for col, name, inverse in ((c1, "Revenue", False), (c2, "Costs", True), (c3, "Operating income", False)):
    a, b = act_m.loc[now, name], bud_m.loc[now, name] if now in bud_m.index else np.nan
    col.metric(
        "Total costs" if name == "Costs" else name,
        money(a, 2),
        f"{signed_money(a - b)} ({(a - b) / abs(b):+.1%})" if pd.notna(b) else None,
        delta_color="inverse" if inverse else "normal",
        border=True,
    )
cur = flags[flags["period"] == str(AS_OF)]
counts = cur["severity"].value_counts()
c4.metric(
    "Red · Amber lines",
    f"{counts.get('Red', 0)} · {counts.get('Amber', 0)}",
    f"of {len(cur)} P&L lines",
    delta_color="off",
    delta_arrow="off",
    border=True,
)

# --- full year landing ------------------------------------------------------------------------
fy_year, n_act = int(outlook["fiscal_year"].iloc[0]), int(outlook["months_actual"].iloc[0])
st.subheader(f"FY{fy_year} landing estimate")
st.caption(f"{n_act} months of actuals + the rolling forecast for the remaining {12 - n_act}, compared with the annual budget.")
fy = outlook.groupby("group")[["fy_outlook_usd", "fy_budget_usd"]].sum()
fy.loc["Operating income"] = fy.loc["Revenue"] - fy.loc["Costs"]
c1, c2, c3, c4 = st.columns(4)
for col, name, inverse in ((c1, "Revenue", False), (c2, "Costs", True), (c3, "Operating income", False)):
    o, b = fy.loc[name, "fy_outlook_usd"], fy.loc[name, "fy_budget_usd"]
    col.metric(
        "Total costs" if name == "Costs" else name,
        money(o, 1),
        f"{signed_money(o - b)} vs budget",
        delta_color="inverse" if inverse else "normal",
        border=True,
    )
c4.metric(
    "Accuracy gain",
    f"+{acc['improvement_vs_budget']:.0%}",
    f"error {acc['rolling_forecast_wape']:.1%} vs {acc['budget_wape']:.1%}",
    delta_color="off",
    delta_arrow="off",
    border=True,
    help="Rolling forecast vs static budget. WAPE = total absolute error / total actuals, scored out of sample across all 27 P&L lines over the last 12 closed months.",
)

# --- trend charts -----------------------------------------------------------------------------
st.subheader("Actuals, budget and rolling forecast")
start = (AS_OF - 23).to_timestamp()
cols = st.columns(2)
for col, name in zip(cols, ("Revenue", "Costs"), strict=True):
    a = act_m.loc[act_m.index >= start, name]
    f = pd.concat([a.iloc[[-1]], fc_m[name]])  # connect the forecast to the last actual
    b = bud_m.loc[bud_m.index >= start, name]
    fig = go.Figure()
    fig.add_scatter(x=a.index, y=a, name="Actual", mode="lines", line=dict(color=p["actual"], width=2), hovertemplate="%{y:$,.0f}")
    fig.add_scatter(x=b.index, y=b, name="Budget", mode="lines", line=dict(color=p["budget"], width=2), hovertemplate="%{y:$,.0f}")
    fig.add_scatter(x=f.index, y=f, name="Rolling forecast", mode="lines", line=dict(color=p["forecast"], width=2, dash="dot"), hovertemplate="%{y:$,.0f}")
    fig.add_vline(x=now, line_width=1, line_color=p["axis"])
    fig.update_yaxes(tickprefix="$", tickformat="~s")
    col.markdown(f"**{'Total costs' if name == 'Costs' else name}**, monthly")
    col.plotly_chart(style_fig(fig, height=330), use_container_width=True, theme=None)

# --- what needs attention ---------------------------------------------------------------------
st.subheader("What needs attention this month")
needs = cur[cur["severity"] != "Green"].copy()
needs["rank"] = needs["severity"].map({"Red": 0, "Amber": 1})
needs = needs.sort_values(["rank", "impact_usd"]).head(5)
if needs.empty:
    st.success("All lines are on track this month.")
for r in needs.itertuples():
    with st.container(border=True):
        st.markdown(md(f"{STATUS_ICON[r.severity]} **{r.severity} · {r.fs_line} ({r.department})** · {signed_money(r.var_budget_usd)} vs budget"))
        st.caption(md(r.commentary))
st.page_link("views/variance.py", label="Open the variance radar for drill-downs", icon="🚦")
