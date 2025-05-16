from dash import html

def render_page2():
    return html.Div([
        html.H3("Welcome to Page 2"),
        html.Div(id="data-check")
    ])
