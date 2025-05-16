import dash
from dash import html, dcc, Output, Input, State
import pandas as pd
import base64
import io

# Importa las páginas
from pages.page1 import render_page1
from pages.page2 import render_page2
from pages.page3 import render_page3
from pages.page4 import render_page4
from pages.page5 import render_page5
from components.sidebar import sidebar

# Inicializa la app
app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server  # si lo despliegas



# Contenido principal
content = html.Div(id="page-content", style={"margin-left": "22%", "padding": "20px"})

# Layout completo
app.layout = html.Div([
    dcc.Location(id="url"),
    dcc.Store(id="stored-data", storage_type="memory"),
    sidebar,
    content
])

# Renderizado por pathname
@app.callback(Output("page-content", "children"), Input("url", "pathname"))
def display_page(pathname):
    if pathname == "/page-1":
        return render_page1()
    elif pathname == "/page-2":
        return render_page2()
    elif pathname == "/page-3":
        return render_page3()
    elif pathname == "/page-4":
        return render_page4()
    elif pathname == "/page-5":
        return render_page5()
    else:
        return html.H3("Welcome! Please choose a page from the sidebar.")

# Carga de CSV
@app.callback(
  Output("stored-data", "data"),
  Output("upload-output", "children"),
  Input("upload-data", "contents"),
  State("upload-data", "filename"),
  prevent_initial_call=True
)
def parse_upload(contents, filename):
  if contents:
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
      df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
      return df.to_json(date_format='iso', orient='split'), html.Div(f"Uploaded {filename} successfully.")
    except Exception as e:
        return None, html.Div(f"Error processing file: {str(e)}")
  return None, html.Div("No file uploaded.")

# Estado del dataframe
@app.callback(
  Output("data-check", "children"),
  Input("stored-data", "data"),
  prevent_initial_call='initial_duplicate'
)

def check_data_status(data):
  if data:
    df = pd.read_json(data, orient='split')
    return f"DataFrame is loaded with {df.shape[0]} rows and {df.shape[1]} columns."
  else:
    return "Primero carga los datos"

if __name__ == "__main__":
  app.run(debug=True)
