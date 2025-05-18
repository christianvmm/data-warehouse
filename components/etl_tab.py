from dash import html, dcc
import pandas as pd
from utils.file_to_df import file_to_df

def etl_tab(filepath):
    if filepath is None:
        return html.Div("No se ha subido ningún archivo.")

    df = file_to_df(filepath)

    return html.Div([
        html.H4("Limpieza de datos"),

        # --------------------------
        html.Div([
            html.Label("1. Eliminar columnas:"),
            dcc.Dropdown(
                id='etl-drop-columns',
                options=[{'label': col, 'value': col} for col in df.columns],
                multi=True
            ),
        ], style={'marginBottom': '20px'}),

        # --------------------------
        html.Div([
            html.Label("2. Convertir columnas de fecha:"),
            dcc.Dropdown(
                id='etl-convert-date',
                options=[{'label': col, 'value': col} for col in df.columns],
                multi=True
            )
        ], style={'marginBottom': '20px'}),
        
        # --------------------------

        html.Div([
            html.Label("Opciones adicionales:"),
            dcc.Checklist(
                id='etl-options',
                options=[
                    {'label': 'Reemplazar valores nulos', 'value': 'nulls'},
                    {'label': 'Eliminar duplicados', 'value': 'duplicates'},
                ],
                value=[],  # Ninguna opción seleccionada por defecto
                labelStyle={'display': 'block'}
            )
        ]),


        # --------------------------

        html.Hr(),
        html.H4("Transformación de datos"),

        html.Div([
            html.Label("1. Columnas a normalizar:"),
            dcc.Dropdown(
                id='etl-normalize-columns',
                options=[{'label': col, 'value': col} for col in df.select_dtypes(include='number').columns],
                multi=True
            )
        ], style={'marginBottom': '20px'}),

         # --------------------------
        html.Div([
            html.Label("2. Filtro por valores numericos"),
            dcc.Dropdown(
                id='etl-filter-column',
                options=[{'label': col, 'value': col} for col in df.select_dtypes(include='number').columns],
                placeholder='Selecciona una columna numérica'
            ),
            dcc.Input(id='etl-filter-min', type='number', placeholder='Valor mínimo'),
            dcc.Input(id='etl-filter-max', type='number', placeholder='Valor máximo'),
        ], style={'marginBottom': '20px'}),

       

        html.Button('Aplicar transformaciones', id='etl-apply-button', n_clicks=0),
        html.Div(id='etl-feedback', style={'marginTop': '10px', 'color': 'green'}),

        html.Hr(),

        html.H5("Vista previa de dataframe transformado:"),
        html.Div(id='etl-table', style={'marginTop': '20px'}),

        dcc.Store(id='transformed-df', storage_type='session'),
        html.Div([
            html.Label("Selecciona formato de descarga:"),
            dcc.RadioItems(
                id='download-format',
                options=[
                    {'label': 'CSV', 'value': 'csv'},
                    {'label': 'JSON', 'value': 'json'},
                    {'label': 'Excel', 'value': 'excel'}
                ],
                value='csv',
                labelStyle={'display': 'inline-block', 'marginRight': '10px'}
            ),
            html.Button("Descargar archivo", id='download-button', n_clicks=0),
            dcc.Download(id="download-data")
        ], style={'marginTop': '30px'}),
    ])
