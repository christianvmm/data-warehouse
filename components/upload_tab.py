from dash import  dcc, html
from utils.read_file import read_file
from dash import dash_table


def upload_tab(filepath):
  children = [
            dcc.Upload(
                id='upload-data',
                children=html.Button('Subir Archivo'),
                multiple=False
            )
        ]

  if filepath:
      try:
          df = read_file(filepath)
          children.append(render_table(df))
      except Exception as e:
          children.append(html.Div(f"Error leyendo archivo: {str(e)}"))

  return children


def render_table(df):
    return dash_table.DataTable(
        df.to_dict('records'),
        [{'name': i, 'id': i} for i in df.columns]
    )