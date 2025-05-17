import os
import uuid
from dash import Dash, dcc, html, dash_table, Input, Output, State, callback, no_update
import datetime
import pandas as pd
import base64
from utils.file_to_df import file_to_df
from utils.read_file import read_file
from utils.save_file import save_file
from components.upload_tab import upload_tab
from components.info_tab import info_tab
from components.etl_tab import etl_tab
from components.mineria_tab import mineria_tab
from components.resultados_tab import resultados_tab
import tempfile

TMP_DIR = os.path.join(tempfile.gettempdir(), 'dash_uploads')
os.makedirs(TMP_DIR, exist_ok=True) # Asegura que exista la carpeta

# TMP_DIR = '/tmp/dash_uploads'
# os.makedirs(TMP_DIR, exist_ok=True)  # Asegura que exista la carpeta


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# Configura límite de subida (opcional, para archivos > 16MB)
app = Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)
app.server.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB

app.layout = html.Div([
    dcc.Store(id='stored-filename'),  # Guardamos solo el nombre del archivo
    
    html.Div([
        html.H3("Sube un archivo (.csv, .xlsx, .json):"),
        dcc.Upload(
            id='upload-data',
            children=html.Button('Subir Archivo'),
            multiple=False
        )
    ], style={'textAlign': 'center', 'marginBottom': '20px'}),

    dcc.Tabs(id='tabs', value='tab-upload', children=[
        dcc.Tab(label='Subir y ver datos', value='tab-upload'),
        dcc.Tab(label='Información', value='tab-info'),
        dcc.Tab(label='ETL', value='tab-etl'),
        dcc.Tab(label='Minería de datos', value='tab-mineria'),
        dcc.Tab(label='Resultados', value='tab-resultados'),
    ]),
    
    html.Div(id='tab-content')
    
])

# @callback(
#     Output('tab-content', 'children'),
#     Input('tabs', 'value'),
#     Input('stored-filename', 'data'),
#     prevent_initial_call=True
# )

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
    print("Archivo guardado en:", filepath)  # 👈 Aquí deberías ver el print
    return filepath

@callback(
    Output('tab-content', 'children'),
    Input('tabs', 'value'),
    Input('stored-filename', 'data'),
)
def render_tab(tab, filepath):
    print("Callback de render_tab activado")
    if tab == 'tab-upload':
        return upload_tab(filepath)
    
    elif tab == 'tab-info':
        return info_tab(filepath)
    
    elif tab == 'tab-etl':
        return etl_tab(filepath)
    
    elif tab == 'tab-mineria':
        return mineria_tab(filepath)
    
    elif tab == 'tab-resultados':
        return resultados_tab(filepath)
    
    
        
if __name__ == '__main__':
    app.run(debug=True)
