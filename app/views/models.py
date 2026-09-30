import numpy as np
import plotly.graph_objects as go
import streamlit as st
from common import as_of, load, palette, style_fig

AS_OF = as_of()
p = palette()
acc = load("accuracy_summary").iloc[0]
byh = load("accuracy_by_horizon")
bys = load("accuracy_by_series")
lb = load("leaderboard")
champs = load("champions")
imp = load("lgb_importance").sort_values("gain_share")

st.title("Model performance")
st.markdown(
    "Every line runs a **tournament** between six forecasting methods. I replay the last 24 month-ends as if each were "
    "the latest close, forecast 12 months ahead, and score the forecasts against what actually happened. For each line the "
    "most accurate method becomes the **champion**, and near-ties go to the simpler method. Everything below is **out of sample**: "
    "no forecast was scored against data it had seen."
)

c = st.columns(4)
c[0].metric("Forecast error", f"{acc['rolling_forecast_wape']:.1%}", "1-12 months ahead", delta_color="off", delta_arrow="off",
            help=f"WAPE over the last 12 closed months ({acc['window_start']} to {acc['window_end']})", border=True)
c[1].metric("Next-month error", f"{acc['rolling_forecast_1m_wape']:.1%}", "1 month ahead", delta_color="off", delta_arrow="off", border=True)
c[2].metric("Budget error", f"{acc['budget_wape']:.1%}", "static annual plan", delta_color="off", delta_arrow="off", border=True)
c[3].metric("Better than budget", f"{acc['improvement_vs_budget']:.0%}", "lower error", delta_color="off", delta_arrow="off", border=True)

left, right = st.columns(2)
with left:
    fig = go.Figure()
    fig.add_bar(x=byh["bucket"], y=byh["rolling_forecast_wape"], name="Rolling forecast", marker_color=p["forecast"], width=0.45,
                text=[f"{v:.1%}" for v in byh["rolling_forecast_wape"]], textposition="outside", hovertemplate="%{y:.1%}")
    fig.add_scatter(x=byh["bucket"], y=byh["budget_wape"], name=f"Static budget ({byh['budget_wape'].iloc[0]:.1%})",
                    mode="lines", line=dict(color=p["budget"], width=2), hovertemplate="%{y:.1%}")
    fig.update_yaxes(tickformat=".0%", range=[0, byh["budget_wape"].max() * 1.4])
    st.markdown("**Error by forecast horizon** (lower is better)")
    st.plotly_chart(style_fig(fig, height=340, hover="closest"), use_container_width=True, theme=None)
with right:
    order = ["Naive", "SeasonalNaive", "ARRRunRate", "AutoETS", "AutoARIMA", "LightGBM"]
    friendly = {"Naive": "Naive", "SeasonalNaive": "Seasonal naive", "ARRRunRate": "ARR run-rate",
                "AutoETS": "Exp. smoothing (ETS)", "AutoARIMA": "ARIMA", "LightGBM": "LightGBM"}
    mix = champs["champion"].value_counts().reindex(order).fillna(0).astype(int)
    fig = go.Figure(go.Bar(x=mix.values, y=[friendly[m] for m in mix.index], orientation="h", marker_color=p["actual"], width=0.55,
                           text=mix.values, textposition="outside", hovertemplate="%{y}: %{x} lines<extra></extra>"))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(showgrid=True, gridcolor=p["grid"], range=[0, mix.max() * 1.2])
    st.markdown("**Champion model by P&L line** (27 lines)")
    st.plotly_chart(style_fig(fig, height=340, legend=False, hover="closest"), use_container_width=True, theme=None)

st.subheader("Line by line: rolling forecast vs budget")
t = bys.merge(champs[["unique_id", "champion"]], on="unique_id").sort_values("actual_12m_usd", ascending=False)
t["Winner"] = np.where(t["rolling_forecast_wape"] < t["budget_wape"], "✅ Forecast", "Budget")
st.dataframe(
    t[["fs_line", "department", "champion", "actual_12m_usd", "rolling_forecast_wape", "budget_wape", "seasonal_naive_wape", "Winner"]],
    hide_index=True, use_container_width=True,
    column_config={
        "fs_line": "P&L line", "department": "Department", "champion": "Champion",
        "actual_12m_usd": st.column_config.NumberColumn("Last 12 months actual", format="dollar"),
        "rolling_forecast_wape": st.column_config.ProgressColumn("Forecast error", format="percent", min_value=0, max_value=0.6),
        "budget_wape": st.column_config.ProgressColumn("Budget error", format="percent", min_value=0, max_value=0.6),
        "seasonal_naive_wape": st.column_config.NumberColumn("Same-month-last-year error", format="percent"),
    },
)
share = t.loc[t["rolling_forecast_wape"] < t["budget_wape"], "actual_12m_usd"].sum() / t["actual_12m_usd"].sum()
st.caption(f"The rolling forecast beats the budget on {(t['rolling_forecast_wape'] < t['budget_wape']).sum()} of {len(t)} lines, covering {share:.0%} of P&L dollars. "
           "Where the budget wins, the lines are small and lumpy (legal fees, recruiting, contractors), and one-offs dominate.")

with st.expander("Full leaderboard: every model on every line (24 origins, all horizons)"):
    w = lb[lb["bucket"] == "All horizons"].pivot(index="unique_id", columns="model", values="wape")[[m for m in order if m in lb["model"].unique()]]
    st.dataframe(w.style.format("{:.1%}", na_rep="-").highlight_min(axis=1, color="rgba(12,163,12,0.25)"), use_container_width=True)

c1, c2 = st.columns(2)
with c1:
    st.subheader("What the ML model pays attention to")
    names = {
        "lag1": "Last month", "lag2": "2 months ago", "lag3": "3 months ago", "lag12": "Same month last year",
        "rolling_mean_lag1_window_size3": "3-month average", "rolling_mean_lag1_window_size6": "6-month average",
        "month": "Calendar month", "hc_growth": "Headcount plan growth",
    }
    fig = go.Figure(go.Bar(x=imp["gain_share"], y=imp["feature"].map(names).fillna(imp["feature"]), orientation="h",
                           marker_color=p["actual"], width=0.55, hovertemplate="%{y}: %{x:.0%}<extra></extra>"))
    fig.update_xaxes(tickformat=".0%", showgrid=True, gridcolor=p["grid"])
    fig.update_yaxes(showgrid=False)
    st.plotly_chart(style_fig(fig, height=300, legend=False, hover="closest"), use_container_width=True, theme=None)
    st.caption("LightGBM feature importance (share of total gain).")
with c2:
    st.subheader("Known limitation")
    st.markdown(
        "Costs are **under-forecast by a few percent**, more so further out. About 40% of that comes from one-off cost spikes "
        "that no model can predict, which is exactly what the variance flags catch. The rest reflects a **hiring acceleration in 2026** that "
        "history-only models lag. LightGBM, which sees the hiring plan, has the smallest bias.\n\n"
        "**Next steps:** a driver-based personnel model (headcount plan × cost per head), overlaying known pricing and hiring "
        "decisions, and a bias correction from the trailing tracking signal."
    )
