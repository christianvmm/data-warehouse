from dash import  dcc, html


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
