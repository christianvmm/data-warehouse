from dash import html, dcc, Input, Output, callback
import pandas as pd

def render_page2():
  return html.Div([
    html.H3("Welcome to Page 2"),
    dcc.Store(id="stored-data", storage_type="session"),  # Must match ID in page1.py
    html.Div(id="data-preview"),
  ])

@callback(
  Output("data-preview", "children"),
  Input("stored-data", "data")
)
def display_data(data):
  if data is None:
    return html.Div("No data available. Please upload a file in Page 1.")

  df = pd.DataFrame(data)
    
  # Display a small preview (e.g., first 5 rows)
  preview = df.head().to_dict("records")
  headers = list(df.columns)

  return html.Table([
    html.Thead(html.Tr([html.Th(col) for col in headers])),
    html.Tbody([
      html.Tr([html.Td(row[col]) for col in headers]) for row in preview
    ])
  ])
