import base64
import io
import dash
from dash import dcc, html, Input, Output, State, ctx
import pandas as pd

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
    html.Div(id="upload-output"),
    dcc.Store(id="stored-data") 
  ])
  



# ======================
# PROCESAMIENTO
# ======================
@dash.callback(
  Output('stored-data', 'data'),
  Output('upload-output', 'children'),
  Input('upload-data', 'contents'),
  State('upload-data', 'filename'),
  prevent_initial_call=True
)
def parse_upload(contents, filename):
  if contents is None:
    return dash.no_update, "No se subió ningún archivo."

  content_type, content_string = contents.split(',')
  decoded = base64.b64decode(content_string)
  
  try:
    if filename.endswith('.csv'):
      df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
    else:
      return dash.no_update, "Formato no soportado"

    # Convertimos el DataFrame a un dict para guardarlo en Store
    return df.to_dict('records'), f"{filename} cargado correctamente."
  
  except Exception as e:
    return dash.no_update, f"Error al procesar el archivo: {e}"