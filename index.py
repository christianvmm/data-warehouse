import os
import io
import uuid
from dash import Dash, dcc, html, dash_table, Input, Output, State, callback, no_update
from dash.exceptions import PreventUpdate
import datetime
import pandas as pd
import base64
import plotly.express as px
from utils.file_to_df import file_to_df
from utils.read_file import read_file
from utils.save_file import save_file
from components.upload_tab import upload_tab
from components.info_tab import info_tab
from components.etl_tab import etl_tab
from components.mineria_tab import mineria_tab
from components.resultados_tab import resultados_tab
import tempfile

from components.clasificacion import mostrar_dropdowns_clasificacion, aplicar_tecnica_clasificacion
from components.regresion import mostrar_dropdowns_regresion, aplicar_tecnica_regresion
from components.cluster import mostrar_dropdowns_cluster, aplicar_kmeans

import dash_bootstrap_components as dbc

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


TMP_DIR = os.path.join(tempfile.gettempdir(), 'dash_uploads')
os.makedirs(TMP_DIR, exist_ok=True) # Asegura que exista la carpeta

# TMP_DIR = '/tmp/dash_uploads'
# os.makedirs(TMP_DIR, exist_ok=True)  # Asegura que exista la carpeta


# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
# external_stylesheets = ['https://cdnjs.cloudflare.com/ajax/libs/materialize/1.0.0/css/materialize.min.css']

# límite de subida (para archivos > 16MB)
app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY], suppress_callback_exceptions=True)
app.server.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB
app.layout = dbc.Container([

    dcc.Store(id='transformed-filepath'),
    dcc.Store(id='stored-filename'),

    dbc.Row([
        dbc.Col(
            html.H3(
                "Hotel Reservations",
                className="text-center text-primary",
                style={'marginBottom': '30px', 'marginTop': '30px'}
            ),
            width=12
        )
    ]),

    dbc.Row([
        dbc.Col(
            dcc.Upload(
                id='upload-data',
                children=dbc.Button('Subir Archivo', color='primary', className='me-2'),
                multiple=False,
                style={'display': 'block', 'margin': '0 auto', 'width': '150px', 'textAlign': 'center'}
            ),
            width=12
        )
    ], justify='center'),

    dbc.Row([
        dbc.Col(
            dcc.Tabs(
                id='tabs',
                value='tab-upload',
                children=[
                    dcc.Tab(label='Subir y ver datos', value='tab-upload'),
                    dcc.Tab(label='Información', value='tab-info'),
                    dcc.Tab(label='ETL', value='tab-etl'),
                    dcc.Tab(label='Minería de datos', value='tab-mineria'),
                    dcc.Tab(label='Resultados', value='tab-resultados'),
                ],
                style={'maxWidth': '1800px', 'margin': '20px auto'}
            ),
            width=12
        )
    ], justify='center'),

    dbc.Row([
        dbc.Col(
            html.Div(id='tab-content', style={'marginTop': '30px', 'maxWidth': '4000px'}),
            width=12
        )
    ])

], fluid=True)
# Callback para guardar el archivo subido
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
    # print("Archivo guardado en:", filepath)
    return filepath




# Callback para ETL
@callback(
    Output('etl-table', 'children'),
    Output('etl-feedback', 'children'),
    Output('transformed-filepath', 'data'),
    Input('etl-apply-button', 'n_clicks'),
    # Limpieza
    State('etl-drop-columns', 'value'),
    State('etl-convert-date', 'value'),
    State('etl-options', 'value'), 
    # Transformacion
    State('etl-normalize-columns', 'value'),
    State('etl-filter-column', 'value'),
    State('etl-filter-min', 'value'),
    State('etl-filter-max', 'value'),
    State('stored-filename', 'data'),
    
    prevent_initial_call=True
)
def aplicar_etl(n_clicks, cols_to_drop, cols_to_date, etl_options,
              normalize_cols, filter_col, filter_min, filter_max, filepath):
    if filepath is None:
        return no_update, "No hay archivo cargado."

    df = file_to_df(filepath)
    feedback_msgs = []

    # 1. Eliminar columnas
    if cols_to_drop:
        df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
        feedback_msgs.append(f"Eliminadas {len(cols_to_drop)} columnas.")

    # 2. Convertir fechas
    if cols_to_date:
        for col in cols_to_date:
            try:
                # Formatos de YYYY/MM/DD o YYYYMMDD, cambian a YYYY-MM-DD
                # YYYY/MM/DD
                df[col] = df[col].astype(str).str.replace('/', '-').str.strip()

                # YYYYMMDD
                def fix_yyyymmdd(fecha):
                    fecha = fecha.strip()
                    if len(fecha) == 8 and fecha.isdigit():
                        return fecha[:4] + '-' + fecha[4:6] + '-' + fecha[6:]
                    return fecha

                df[col] = df[col].apply(fix_yyyymmdd)

                df[col] = pd.to_datetime(df[col], errors='coerce')
                df[col] = df[col].dt.strftime('%Y-%m-%d')
                feedback_msgs.append(f"'{col}' convertido a formato YYYY-MM-DD.")
            except Exception as e:
                feedback_msgs.append(f"Error al convertir '{col}' a fecha: {str(e)}")

    # 3. Reemplazar nulos 
    if 'nulls' in etl_options:
        for col in df.columns:
            if df[col].isnull().any():
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col].fillna(0, inplace=True)
                    feedback_msgs.append(f"Nulos en '{col}' reemplazados por 0.")
                else:
                    df[col].fillna('Desconocido', inplace=True)
                    feedback_msgs.append(f"Nulos en '{col}' reemplazados por 'Desconocido'.")

    # 4. Eliminar duplicados
    if 'duplicates' in etl_options:
        before = len(df)
        df.drop_duplicates(inplace=True)
        after = len(df)
        removed = before - after
        feedback_msgs.append(f"Eliminadas {removed} filas duplicadas.")


     # 5. Normalización
    if normalize_cols:
        for col in normalize_cols:
            if pd.api.types.is_numeric_dtype(df[col]):
                min_val = df[col].min()
                max_val = df[col].max()
                if max_val != min_val:
                    df[col] = (df[col] - min_val) / (max_val - min_val)
                    feedback_msgs.append(f"Columna '{col}' normalizada.")
                else:
                    feedback_msgs.append(f"No se normalizó '{col}' (valor constante).")

    # 6 . Filtrado 
    if filter_col and (filter_min is not None or filter_max is not None):
        original_len = len(df)
        if filter_min is not None:
            df = df[df[filter_col] >= filter_min]
        if filter_max is not None:
            df = df[df[filter_col] <= filter_max]
        feedback_msgs.append(f"Filtrado aplicado en '{filter_col}'. Filas reducidas de {original_len} a {len(df)}.")

    # Guardar el dataframe transformado en un archivo temporal
    filename = f"processed_{uuid.uuid4().hex}.csv"
    fullpath = os.path.join(TMP_DIR, filename)
    df.to_csv(fullpath, index=False)

    # Mostrar tabla
    table = dash_table.DataTable(
        data=df.head(20).to_dict('records'),
        columns=[{'name': i, 'id': i} for i in df.columns],
        style_table={'overflowX': 'auto'},
        style_cell={'textAlign': 'left'}
    )

    # feedback = f"Archivo procesado guardado: {filename}"

    # return table, html.Ul([html.Li(msg) for msg in feedback_msgs]), df.to_json(date_format='iso', orient='split')
    # return (
    #     table,
    #     html.Ul([html.Li(msg) for msg in feedback_msgs]),
    #     {
    #         'columns': df.columns.tolist(),
    #         'data': df.to_dict('records')
    #     }
    # )
    return table, html.Ul([html.Li(msg) for msg in feedback_msgs]), filename

# Callback para descargar el archivo transformado
@callback(
    Output("download-data", "data"),
    Input("download-button", "n_clicks"),
    State("transformed-df", "data"),
    State("download-format", "value"),
    prevent_initial_call=True
)
def download_file(n_clicks, df_json, format_selected):
    if df_json is None:
        raise PreventUpdate

    df = pd.read_json(df_json, orient='split')

    if format_selected == "csv":
        return dcc.send_bytes(df.to_csv(index=False).encode(), "datos_transformados.csv")

    elif format_selected == "json":
        return dcc.send_bytes(df.to_json(orient="records", indent=2).encode(), "datos_transformados.json")

    elif format_selected == "excel":
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Datos')
        buffer.seek(0)
        return dcc.send_bytes(buffer.read(), "datos_transformados.xlsx")

    else:
        raise PreventUpdate


# Callback para graficos
@callback(
    Output('eda-plots-container', 'children'),
    Input('eda-numeric-dropdown', 'value'),
    State('transformed-filepath', 'data'),
    prevent_initial_call=True
)
def update_eda_graficos(selected_col, processed_filename):
    if selected_col is None or processed_filename is None:
        return html.Div("No hay columna seleccionada o datos procesados.")

    fullpath = os.path.join(TMP_DIR, processed_filename)
    if not os.path.exists(fullpath):
        return html.Div("Archivo procesado no encontrado.")

    df = pd.read_csv(fullpath)

    fig_hist = px.histogram(df, x=selected_col, nbins=30, title=f"Histograma de Columna '{selected_col}'")
    fig_box = px.box(df, y=selected_col, title=f"Boxplot de Columna '{selected_col}'")

    return html.Div([
        dcc.Graph(figure=fig_hist),
        dcc.Graph(figure=fig_box)
    ])


# Callback para renderizar el contenido de la pestaña seleccionada
@callback(
    Output('tab-content', 'children'),
    Input('tabs', 'value'),
    Input('stored-filename', 'data'),
    State('transformed-filepath', 'data')
)
def render_tab(tab, filepath, processed_filename):
    # print("Callback de render_tab activado")
    if tab == 'tab-upload':
        return upload_tab(filepath)
    
    elif tab == 'tab-info':
        return info_tab(filepath)
    
    elif tab == 'tab-etl':
        return etl_tab(filepath)
    
    elif tab == 'tab-mineria':
        # print("DEBUG: df_dict recibido en minería =", df_dict) 
        print("DEBUG - Archivo procesado que se pasa:", processed_filename)
        return mineria_tab(processed_filename)
    
    elif tab == 'tab-resultados':
        return resultados_tab(filepath)
    

if __name__ == '__main__':
    app.run(debug=True)