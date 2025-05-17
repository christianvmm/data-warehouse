import os
import uuid

from dash import Dash, dcc, html, dash_table, Input, Output, State, callback, no_update
import datetime
import pandas as pd
import base64
from utils.file_to_df import file_to_df

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)

# Configura límite de subida (opcional, para archivos > 16MB)
app.server.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB

TMP_DIR = '/tmp/dash_uploads'
os.makedirs(TMP_DIR, exist_ok=True)  # Asegura que exista la carpeta

app.layout = html.Div([
    dcc.Store(id='stored-filename'),  # Guardamos solo el nombre del archivo
    
    dcc.Tabs(id='tabs', value='tab-upload', children=[
        dcc.Tab(label='Subir y ver datos', value='tab-upload'),
        dcc.Tab(label='Información del DataFrame', value='tab-info'),
    ]),
    
    html.Div(id='tab-content')
])

def save_file(contents, filename):
    # contents viene en base64: "data:;base64,ABC123..."
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    
    # Para evitar colisiones, nombre único
    unique_filename = f"{uuid.uuid4()}_{filename}"
    filepath = os.path.join(TMP_DIR, unique_filename)
    
    with open(filepath, "wb") as f:
        f.write(decoded)
    return filepath

def read_file(filepath):
    # Aquí lees el archivo guardado y conviertes a DataFrame
    # Asumiendo que file_to_df pueda leer desde filepath o adaptas:
    with open(filepath, "rb") as f:
        content = f.read()
    # file_to_df espera contenido en base64 + nombre + fecha?
    # Entonces recreamos contenido base64 para usar file_to_df
    b64_content = "data:;base64," + base64.b64encode(content).decode()
    
    filename = os.path.basename(filepath).split("_",1)[1]  # quitar uuid
    date = os.path.getmtime(filepath)
    df = file_to_df(b64_content, filename, date)
    return df

def render_table(df):
    return dash_table.DataTable(
        df.to_dict('records'),
        [{'name': i, 'id': i} for i in df.columns]
    )

@callback(
    Output('stored-filename', 'data'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),
    prevent_initial_call=True
)
def save_uploaded_file(contents, filename):
    if contents is None:
        return no_update
    filepath = save_file(contents, filename)
    # Guardamos solo la ruta del archivo (puedes guardar solo el nombre si prefieres)
    return filepath

@callback(
    Output('tab-content', 'children'),
    Input('tabs', 'value'),
    State('stored-filename', 'data')
)
def render_tab(tab, filepath):
    if tab == 'tab-upload':
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
    
    elif tab == 'tab-info':
        if not filepath:
            return html.Div("No hay datos cargados aún.")
        try:
            df = read_file(filepath)
            return html.Div([
                html.H5(f"Tamaño del DataFrame: {df.shape[0]} filas x {df.shape[1]} columnas")
            ])
        except Exception as e:
            return html.Div(f"Error leyendo archivo: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
