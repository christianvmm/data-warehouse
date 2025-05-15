import dash
from dash import html, dcc, Output, Input

# Initialize the Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)

# Define sidebar links
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

# Define the main content area
content = html.Div(id="page-content", style={"margin-left": "22%", "padding": "20px"})

# Define the full layout
app.layout = html.Div([
    dcc.Location(id="url"),
    sidebar,
    content
])

# Callback to update page content based on URL
@app.callback(Output("page-content", "children"), Input("url", "pathname"))
def display_page(pathname):
    if pathname == "/page-1":
        return html.H3("Welcome to Page 1")
    elif pathname == "/page-2":
        return html.H3("Welcome to Page 2")
    elif pathname == "/page-3":
        return html.H3("Welcome to Page 3")
    elif pathname == "/page-4":
        return html.H3("Welcome to Page 4")
    elif pathname == "/page-5":
        return html.H3("Welcome to Page 5")
    else:
        return html.H3("Welcome! Please choose a page from the sidebar.")

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
