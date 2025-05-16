from dash import html, dcc

def render_page1():
  return html.Div([
    html.H3("Adjunta los datos"),

    dcc.Upload(
      id="upload-data",
      children=html.Div(["Arrastra y suelta o ", html.A("Selecciona un Archivo")]),
      style={
        "width": "100%",
        "height": "60px",
        "lineHeight": "60px",
        "borderWidth": "1px",
        "borderStyle": "dashed",
        "borderRadius": "5px",
        "textAlign": "center",
        "margin": "10px"
      },
      multiple=False
    ),
    html.Div(id="upload-output")
  ])
