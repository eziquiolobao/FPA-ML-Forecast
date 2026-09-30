import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from common import as_of, load, money, palette, rgba, style_fig

AS_OF = as_of()
p = palette()

vint = load("forecast_vintages")
meta = load("series_meta").sort_values(["account_type", "fs_line", "department"], ascending=[False, True, True])
pnl = load("fct_monthly_pnl")
bud = load("fct_budget_monthly")
champs = load("champions").set_index("unique_id")
for d in (pnl, bud):
    d["unique_id"] = d["fs_line"] + " | " + d["department"]
    d["ds"] = pd.PeriodIndex(d["period"], freq="M").to_timestamp()

st.title("Rolling forecast")
st.caption(
    "Pick a P&L line to see its actuals, budget and 12-month rolling forecast. Move the slider to see the forecast "
    "made at an earlier month-end (a *vintage*) and how it compared with what actually happened."
)

c1, c2 = st.columns([1, 1])
lines = meta["fs_line"].unique().tolist()
fs_line = c1.selectbox("P&L line", lines, index=lines.index("Subscription Revenue"))
depts = meta.loc[meta["fs_line"] == fs_line, "department"].tolist()
dept = c2.selectbox("Department", depts)
uid = f"{fs_line} | {dept}"

origins = sorted(vint.loc[vint["is_champion"], "origin"].unique())
labels = [pd.Timestamp(o).strftime("%b-%y") for o in origins]
pick = st.select_slider("Forecast made at month-end", options=labels, value=labels[-1])
origin = origins[labels.index(pick)]

v = vint[(vint["unique_id"] == uid) & (vint["origin"] == origin) & vint["is_champion"]].sort_values("ds")
a = pnl[(pnl["unique_id"] == uid) & (pnl["ds"] >= (AS_OF - 30).to_timestamp())].groupby("ds")["actual_usd"].sum()
b = bud[(bud["unique_id"] == uid) & (bud["ds"] >= (AS_OF - 30).to_timestamp())].groupby("ds")["budget_usd"].sum()
model = v["model"].iloc[0] if len(v) else "-"

FRIENDLY = {
    "Naive": "Naive (last month)", "SeasonalNaive": "Seasonal naive (same month last year)", "ARRRunRate": "ARR run-rate (driver)",
    "AutoETS": "Exponential smoothing (ETS)", "AutoARIMA": "ARIMA", "LightGBM": "LightGBM (machine learning)",
}
st.markdown(f"Champion model for this line: **{FRIENDLY.get(model, model)}**")
m1, m2, m3 = st.columns(3)
m1.metric(
    "Backtest error (WAPE)",
    f"{champs.loc[uid, 'champion_wape']:.1%}" if uid in champs.index else "-",
    border=True,
    help="Average absolute error of the current champion over 24 replayed month-ends, 1-12 months ahead.",
)
nxt = v.head(3)
bud_next = b.reindex(nxt["ds"]).sum(min_count=1)
m2.metric("Next 3 months forecast", money(nxt["yhat"].sum()), f"{money(nxt['yhat'].sum() - bud_next)} vs budget" if pd.notna(bud_next) else None, delta_color="off", border=True)
realised = v.dropna(subset=["y"])
m3.metric(
    "This vintage's error so far",
    f"{(realised['y'] - realised['yhat']).abs().sum() / realised['y'].abs().sum():.1%}" if len(realised) else "n/a",
    f"{len(realised)} month(s) closed since" if len(realised) else "no months closed yet",
    delta_color="off",
    delta_arrow="off",
    border=True,
)

fig = go.Figure()
fig.add_scatter(x=v["ds"], y=v["hi95"], mode="lines", line=dict(width=0), showlegend=False, hovertemplate="high %{y:$,.0f}")
fig.add_scatter(
    x=v["ds"], y=v["lo95"], mode="lines", fill="tonexty", fillcolor=rgba(p["forecast"], p["band_alpha"]), line=dict(width=0),
    name="Expected range (95%)", hovertemplate="low %{y:$,.0f}",
)
fig.add_scatter(x=a.index, y=a, name="Actual", line=dict(color=p["actual"], width=2), mode="lines+markers", marker=dict(size=6), hovertemplate="%{y:$,.0f}")
fig.add_scatter(x=b.index, y=b, name="Budget", mode="lines", line=dict(color=p["budget"], width=2), hovertemplate="%{y:$,.0f}")
fig.add_scatter(
    x=v["ds"], y=v["yhat"], mode="lines", name=f"Forecast made {pick}",
    line=dict(color=p["forecast"], width=2, dash="dot"), hovertemplate="%{y:$,.0f}",
)
fig.add_vline(x=pd.Timestamp(origin), line_width=1, line_color=p["axis"])
fig.update_yaxes(tickprefix="$", tickformat="~s")
st.plotly_chart(style_fig(fig, height=430), use_container_width=True, theme=None)

with st.expander("Forecast table"):
    t = v[["ds", "horizon", "yhat", "lo95", "hi95", "y"]].merge(b.rename("budget").reset_index(), on="ds", how="left")
    t.columns = ["Month", "Months ahead", "Forecast", "Expected low", "Expected high", "Actual", "Budget"]
    t["Month"] = t["Month"].dt.strftime("%b-%y")
    money_cols = {c: st.column_config.NumberColumn(c, format="dollar") for c in ["Forecast", "Expected low", "Expected high", "Actual", "Budget"]}
    st.dataframe(t, hide_index=True, use_container_width=True, column_config=money_cols)
    st.download_button("Download CSV", t.to_csv(index=False), file_name=f"forecast_{uid.replace(' | ', '_')}_{pick}.csv")

st.caption(
    "The expected range is built from this model's own past errors at each horizon, measured against actuals "
    "with one-off spikes removed, so it describes normal business variation."
)
