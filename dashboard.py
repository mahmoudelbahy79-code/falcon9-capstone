from pathlib import Path
import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html, Input, Output

ROOT = Path(__file__).resolve().parent
DF = pd.read_csv(ROOT / "data" / "falcon9_labeled.csv", parse_dates=["Date"])
SITES = sorted(DF["LaunchSite"].dropna().unique())
COLORS = {0: "#B388FF", 1: "#00E5FF"}
app = Dash(__name__)
app.title = "Falcon 9 Landing Analysis"
app.layout = html.Div(style={"fontFamily": "Arial, sans-serif", "background": "#0B1529", "color": "#FFFFFF", "minHeight": "100vh", "padding": "28px"}, children=[
    html.H1("Falcon 9 First-Stage Landing Analysis", style={"marginBottom": "4px"}),
    html.P("Historical capstone sample: 2010–2020 | Source: IBM Skills Network", style={"color": "#94A3B8"}),
    html.Label("Launch site", style={"display": "block", "marginTop": "20px", "marginBottom": "8px"}),
    dcc.Dropdown(id="site", options=[{"label": "All sites", "value": "ALL"}] + [{"label": s, "value": s} for s in SITES], value="ALL", clearable=False,
                 style={"color": "#172235", "maxWidth": "520px"}),
    html.Div(id="kpis", style={"display": "flex", "gap": "18px", "margin": "22px 0"}),
    html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "18px"}, children=[
        dcc.Graph(id="year-chart", config={"displayModeBar": False}),
        dcc.Graph(id="payload-chart", config={"displayModeBar": False}),
    ]),
    html.P("Target definition: Class=1 is the dataset's successful landing label; Class=0 combines other coded outcomes, including no attempt or unspecified results. Descriptive analysis only—not an operational decision tool.", style={"color": "#94A3B8", "fontSize": "13px", "marginTop": "16px"})
])

@app.callback(Output("kpis", "children"), Output("year-chart", "figure"), Output("payload-chart", "figure"), Input("site", "value"))
def update(site):
    d = DF.copy()
    if site != "ALL":
        d = d[d.LaunchSite == site]
    total = len(d)
    rate = d.Class.mean() * 100 if total else 0
    kpis = [
        html.Div([html.Div("Launches", style={"color": "#94A3B8"}), html.H2(f"{total}")], style={"background": "#1A2740", "padding": "16px 22px", "minWidth": "160px"}),
        html.Div([html.Div("Landing label rate", style={"color": "#94A3B8"}), html.H2(f"{rate:.1f}%", style={"color": "#00E5FF"})], style={"background": "#1A2740", "padding": "16px 22px", "minWidth": "160px"}),
    ]
    y = d.groupby(d.Date.dt.year).Class.agg(["mean", "size"]).reset_index().rename(columns={"Date": "Year", "mean": "Rate", "size": "Launches"})
    fig1 = px.line(y, x="Date", y="Rate", markers=True, hover_data={"Launches": True}, title="Landing-label rate by year", labels={"Date": "Year", "Rate": "Rate"})
    fig1.update_yaxes(tickformat=".0%", range=[0, 1])
    fig1.update_layout(paper_bgcolor="#0B1529", plot_bgcolor="#0B1529", font_color="#FFFFFF", margin=dict(l=20, r=20, t=55, b=20))
    fig2 = px.scatter(d, x="PayloadMass", y="FlightNumber", color=d.Class.astype(str), color_discrete_map={"0": COLORS[0], "1": COLORS[1]},
        hover_data=["Date", "LaunchSite", "Orbit", "Outcome"], title="Payload mass vs. flight number and landing label",
        labels={"PayloadMass": "Payload mass (kg)", "FlightNumber": "Flight number", "color": "Class"})
    fig2.update_layout(paper_bgcolor="#0B1529", plot_bgcolor="#0B1529", font_color="#FFFFFF", margin=dict(l=20, r=20, t=55, b=20))
    return kpis, fig1, fig2

if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=8050)
