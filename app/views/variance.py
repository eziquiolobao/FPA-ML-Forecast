import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from common import (
    REPORTS,
    STATUS_COLOR,
    STATUS_ICON,
    STATUS_LABEL,
    as_of,
    load,
    md,
    money,
    palette,
    rgba,
    run_metadata,
    signed_money,
    style_fig,
)

AS_OF = as_of()
p = palette()
flags = load("variance_monthly")
detail = load("gl_detail_recent")

st.title("Variance radar")
st.caption(
    "🔴 **Red**: material vs budget *and* outside the range the forecast expected, so investigate first.  "
    "🟠 **Amber**: material but expected, a small surprise, or drifting the same way for 3+ months, so explain or monitor.  "
    "🟢 **Green**: on track."
)

periods = sorted(flags["period"].unique(), reverse=True)
c1, c2, c3 = st.columns([1, 1, 2])
period = c1.selectbox("Month", periods, format_func=lambda x: pd.Period(x, "M").strftime("%B %Y"))
show = c2.multiselect("Status", ["Red", "Amber", "Green"], default=["Red", "Amber"])
depts = sorted(flags["department"].unique())
dept_pick = c3.multiselect("Department", depts, placeholder="All departments")

cur = flags[flags["period"] == period]
counts = cur["severity"].value_counts()
m = st.columns(3)
for col, s in zip(m, ("Red", "Amber", "Green"), strict=True):
    col.metric(f"{STATUS_ICON[s]} {s} · {STATUS_LABEL[s]}", int(counts.get(s, 0)), border=True)

view = cur[cur["severity"].isin(show)]
if dept_pick:
    view = view[view["department"].isin(dept_pick)]
view = view.assign(rank=view["severity"].map({"Red": 0, "Amber": 1, "Green": 2})).sort_values(["rank", "impact_usd"])
table = pd.DataFrame(
    {
        "Status": view["severity"].map(lambda s: f"{STATUS_ICON[s]} {s}"),
        "Department": view["department"],
        "P&L line": view["fs_line"],
        "Actual": view["actual_usd"],
        "Budget": view["budget_usd"],
        "Var vs budget": view["var_budget_usd"],
        "Var %": view["var_budget_pct"] * 100,
        "Outside expected range": view["outside_range"],
        "Months drifting": view["drift_months"],
    }
)
st.markdown("**Select a row to drill down.**")
event = st.dataframe(
    table,
    hide_index=True,
    use_container_width=True,
    on_select="rerun",
    selection_mode="single-row",
    column_config={
        "Actual": st.column_config.NumberColumn(format="dollar"),
        "Budget": st.column_config.NumberColumn(format="dollar"),
        "Var vs budget": st.column_config.NumberColumn(format="dollar"),
        "Var %": st.column_config.NumberColumn(format="%+.1f%%"),
    },
)

rows = event.selection.rows if event and event.selection else []
if rows or len(view):
    r = view.iloc[rows[0] if rows else 0]
    st.divider()
    st.subheader(f"{STATUS_ICON[r['severity']]} {r['fs_line']} · {r['department']} · {pd.Period(r['period'], 'M').strftime('%b %Y')}")
    if not rows:
        st.caption("Showing the first row. Select another row in the table above to change it.")
    left, right = st.columns([3, 2])
    with left:
        st.markdown("**Draft commentary**")
        st.info(md(r["commentary"]) or "On track, no commentary needed.")
        k = st.columns(3)
        k[0].metric("Actual", money(r["actual_usd"]), f"{signed_money(r['var_budget_usd'])} vs budget", delta_color="normal" if r["account_type"] == "Revenue" else "inverse")
        k[1].metric("Last month's forecast", money(r["forecast_usd"]), f"{signed_money(r['var_forecast_usd'])} surprise" if pd.notna(r["var_forecast_usd"]) else None, delta_color="off")
        k[2].metric("Prior year", money(r["prior_year_usd"]), f"{r['var_yoy_pct']:+.0%} YoY" if pd.notna(r["var_yoy_pct"]) else None, delta_color="off")
        if r["personnel_split"]:
            s = json.loads(r["personnel_split"])
            st.markdown(md(
                f"**Headcount vs rate:** {int(s['hc_actual'])} heads vs {int(s['hc_plan'])} planned "
                f"→ volume {signed_money(s['volume_usd'])}, cost per head {signed_money(s['rate_usd'])}"
            ))
        drivers = pd.DataFrame(json.loads(r["drivers"]))
        if not drivers.empty:
            st.markdown("**What moved vs a typical recent month** (median of the prior 3 months)")
            names = {"driver": "Driver", "current_usd": "This month", "typical_usd": "Typical month", "change_usd": "Change", "documents": "Documents"}
            drivers = drivers[list(names)].rename(columns=names)
            st.dataframe(
                drivers, hide_index=True, use_container_width=True,
                column_config={c: st.column_config.NumberColumn(format="dollar") for c in ("This month", "Typical month", "Change")},
            )
    with right:
        hist = flags[flags["unique_id"] == r["unique_id"]].sort_values("period")
        x = pd.PeriodIndex(hist["period"], freq="M").to_timestamp()
        fig = go.Figure()
        fig.add_scatter(x=x, y=hist["range_hi_usd"], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip")
        fig.add_scatter(x=x, y=hist["range_lo_usd"], mode="lines", fill="tonexty", fillcolor=rgba(p["forecast"], p["band_alpha"]), line=dict(width=0), name="Expected range", hoverinfo="skip")
        fig.add_scatter(x=x, y=hist["budget_usd"], name="Budget", mode="lines", line=dict(color=p["budget"], width=2), hovertemplate="%{y:$,.0f}")
        fig.add_scatter(x=x, y=hist["actual_usd"], name="Actual", mode="lines+markers", line=dict(color=p["actual"], width=2), hovertemplate="%{y:$,.0f}",
                        marker=dict(size=9, color=[STATUS_COLOR[s] for s in hist["severity"]], line=dict(width=2, color="rgba(0,0,0,0)")))
        fig.update_yaxes(tickprefix="$", tickformat="~s")
        st.markdown("**Last 12 months** (markers colored by status)")
        st.plotly_chart(style_fig(fig, height=330), use_container_width=True, theme=None)

    gl = detail[(detail["fs_line"] == r["fs_line"]) & (detail["department"] == r["department"]) & (detail["period"] == r["period"])]
    with st.expander(f"Journal-entry lines behind this number ({len(gl):,})"):
        gl = gl.reindex(gl["amount_usd"].abs().sort_values(ascending=False).index)
        st.dataframe(
            gl[["je_id", "posting_date", "entity", "account_name", "cost_center_name", "vendor_name", "document_number", "description", "source", "amount_usd"]],
            hide_index=True, use_container_width=True, column_config={"amount_usd": st.column_config.NumberColumn("Amount (USD)", format="dollar")},
        )

st.divider()
st.subheader("Twelve months at a glance")
code = {"Green": 0, "Amber": 1, "Red": 2}
grid = flags.pivot(index="unique_id", columns="period", values="severity")
grid = grid.loc[flags.groupby("unique_id")["severity"].apply(lambda s: s.map(code).sum()).sort_values().index]
fig = go.Figure(
    go.Heatmap(
        z=grid.apply(lambda c: c.map(code)).to_numpy(),
        x=[pd.Period(c, "M").strftime("%b-%y") for c in grid.columns],
        y=grid.index,
        text=grid.to_numpy(),
        hovertemplate="%{y}<br>%{x}: %{text}<extra></extra>",
        colorscale=[[0, p["neutral"]], [0.33, p["neutral"]], [0.34, STATUS_COLOR["Amber"]], [0.66, STATUS_COLOR["Amber"]], [0.67, STATUS_COLOR["Red"]], [1, STATUS_COLOR["Red"]]],
        zmin=0, zmax=2, showscale=False, xgap=2, ygap=2,
    )
)
st.plotly_chart(style_fig(fig, height=720, legend=False, hover="closest"), use_container_width=True, theme=None)
st.caption("Legend: 🔴 Red · 🟠 Amber · blank = 🟢 Green. Hover a cell for details.")

st.divider()
c1, c2 = st.columns([2, 3])
with c1:
    st.subheader("Monthly variance pack")
    pack = REPORTS / f"variance_pack_{AS_OF}.xlsx"
    if pack.exists():
        st.download_button(
            f"⬇️ Download Excel pack ({AS_OF.strftime('%b %Y')})", pack.read_bytes(), file_name=pack.name,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
        st.caption("Summary tab, one tab per department with month and YTD variances, and the full-year outlook.")
with c2:
    ev = run_metadata().get("variance", {}).get("evaluation", {})
    if ev:
        st.subheader("Does the flagging work?")
        st.markdown(
            f"The data generator secretly injected **{ev['anomalies_in_window']} accounting problems** "
            f"(duplicate invoices, missed accruals, runaway cloud spend…) over {ev['window'].replace('..', ' to ')}. "
            f"Red flags caught **{ev['recall_red']:.0%}** of them, and **{ev['precision_red']:.0%}** of Red flags were real problems."
        )
        with st.expander("See every injected anomaly and how it was flagged"):
            e = load("flag_evaluation")
            e["best_flag"] = e["best_flag"].map(lambda s: f"{STATUS_ICON[s]} {s}")
            st.dataframe(
                e[["period", "type", "description", "series", "approx_usd", "best_flag"]], hide_index=True, use_container_width=True,
                column_config={"approx_usd": st.column_config.NumberColumn("Approx. size", format="dollar")},
            )
