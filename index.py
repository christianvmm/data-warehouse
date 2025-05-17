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


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# Configura límite de subida (opcional, para archivos > 16MB)
app = Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)
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
      return upload_tab(filepath)
    
    elif tab == 'tab-info':
      return info_tab(filepath)
        
if __name__ == '__main__':
    app.run(debug=True)
