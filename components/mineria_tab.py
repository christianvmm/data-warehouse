import os
import tempfile
import pandas as pd
from dash import html, dcc, dash_table
import dash_bootstrap_components as dbc
from components.k_means_clustering_component import k_means_clustering_component
from components.otro_metodo_component import otro_metodo_component

TMP_DIR = os.path.join(tempfile.gettempdir(), 'dash_uploads')

def mineria_tab(processed_filename):
    if processed_filename is None:
        return dbc.Container([
            dbc.Alert("No hay datos procesados desde ETL.", color="secondary", className="mt-4")
        ], fluid=True)

    fullpath = os.path.join(TMP_DIR, processed_filename)
    if not os.path.exists(fullpath):
        return dbc.Container([
            dbc.Alert("Archivo procesado no encontrado.", color="danger", className="mt-4")
        ], fluid=True)

    df = pd.read_csv(fullpath)
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    if not numeric_cols:
        return dbc.Container([
            dbc.Alert("No se encontraron columnas numéricas.", color="warning", className="mt-4")
        ], fluid=True)

    
    numeric_df = df[numeric_cols]
    stats_df = numeric_df.describe().T
    stats_df["Mediana"] = numeric_df.median()
    stats_df.rename(columns={
        'count': 'Cantidad',
        'mean': 'Media',
        'std': 'Desviación Estándar',
        'min': 'Mínimo',
        'max': 'Máximo'
    }, inplace=True)
    stats_df.reset_index(inplace=True)
    stats_df.rename(columns={'index': 'Variable'}, inplace=True)

    stats_table = dash_table.DataTable(
        data=stats_df.to_dict('records'),
        columns=[{"name": col, "id": col} for col in stats_df.columns],
        page_size=10,
        style_table={'overflowX': 'auto'},
        style_header={'backgroundColor': '#1e3a8a', 'color': 'white'},
        style_cell={'padding': '5px', 'textAlign': 'left'}
    )

    return dbc.Container(fluid=True, style={'paddingTop': '30px', 'paddingBottom': '30px'}, children=[
        dbc.Row(dbc.Col(html.H4("Exploración y Minería de Datos", className="text-center text-primary mb-4"))),

        dbc.Row([
            dbc.Col(html.P(f"Dimensiones del dataset: {df.shape[0]} filas × {df.shape[1]} columnas."),
                    width=12, className="mb-4")
        ]),

        
        dbc.Card([
            dbc.CardHeader(html.H5("Estadísticas Descriptivas de Columnas Numéricas")),
            dbc.CardBody(stats_table)
        ], className="mb-4 shadow-sm"),

        
        dbc.Card([
            dbc.CardHeader(html.H5("Visualización de Datos Numéricos")),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.Label("Selecciona una columna numérica:", className="form-label"),
                        dcc.Dropdown(
                            id='eda-numeric-dropdown',
                            options=[{'label': col, 'value': col} for col in numeric_cols],
                            value=numeric_cols[0],
                            clearable=False
                        )
                    ], width=6)
                ], className="mb-3"),
                html.Div(id='eda-plots-container')
            ])
        ], className="mb-4 shadow-sm"),

        
        k_means_clustering_component(fullpath),
        otro_metodo_component(fullpath), 

        dcc.Store(id='transformed-filepath', data=processed_filename)
    ])

