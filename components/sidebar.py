from dash import html, dcc

sidebar = html.Div(
  [
    html.H2("Funciones", className="sidebar-title"),
    html.Hr(),
    dcc.Link("Page 1", href="/page-1", className="sidebar-link"),
    html.Br(),
    dcc.Link("Page 2", href="/page-2", className="sidebar-link"),
    html.Br(),
    dcc.Link("Page 3", href="/page-3", className="sidebar-link"),
    html.Br(),
    dcc.Link("Page 4", href="/page-4", className="sidebar-link"),
    html.Br(),
    dcc.Link("Page 5", href="/page-5", className="sidebar-link"),
  ],
  style={
    "padding": "20px",
    "width": "20%",
    "height": "100vh",
    "position": "fixed",
    "top": 0,
    "left": 0,
    "background-color": "#f8f9fa"
  },
)