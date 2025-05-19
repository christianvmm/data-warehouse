import pandas as pd
from dash import html, dcc, dash_table
import os
import tempfile

TMP_DIR = os.path.join(tempfile.gettempdir(), 'dash_uploads')

def mineria_tab(processed_filename):
    if processed_filename is None:
        return html.Div("No hay datos procesados desde ETL.")

    fullpath = os.path.join(TMP_DIR, processed_filename)
    if not os.path.exists(fullpath):
        return html.Div("Archivo procesado no encontrado.")

    df = pd.read_csv(fullpath)

    numeric_cols = df.select_dtypes(include='number').columns.tolist()

    
    if not numeric_cols:
        return html.Div("No se encontraron columnas numéricas.")

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
        style_table={'overflowX': 'auto'},
        style_cell={'textAlign': 'left'},
        page_size=10
    )

    return html.Div([
        dcc.Store(id='transformed-filepath', data=processed_filename), 
        html.H4("Exploración y Minería de Datos"),
        html.P(f"Dimensiones del dataset: {df.shape[0]} filas y {df.shape[1]} columnas."),

        html.H5("Estadísticas Descriptivas de las Columnas Numéricas"),
        stats_table,

        html.Hr(),

        html.H5("Visualización de Datos Numéricos"),
        html.P("Selecciona una columna numérica:"),

        dcc.Dropdown(
            id='eda-numeric-dropdown',
            options=[{'label': col, 'value': col} for col in numeric_cols],
            value=numeric_cols[0], 
            clearable=False,
            placeholder="Selecciona una columna",
        ),

        html.Div(id='eda-plots-container'), 
        html.Hr(),

        html.H5("Técnicas de Minería de Datos"),
        html.P("Técnica a aplicar:"),
        dcc.Dropdown(
            id='mining-technique-dropdown',
            options=[
                {'label': 'Clustering - K-Means', 'value': 'kmeans'},
                {'label': 'Clasificación - Árbol de Decisión', 'value': 'decision_tree'},
                {'label': 'Regresión Lineal', 'value': 'regression'}
            ],
            placeholder="Selecciona una técnica de minería",
        ),

        html.Div(id='cluster-variable-selectors'),

        html.Div(id='mining-output-container'),
    ])
