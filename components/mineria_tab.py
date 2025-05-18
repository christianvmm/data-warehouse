from dash import html
import pandas as pd
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

    return html.Div([
        html.H4("Exploración y Minería de Datos"),
        html.P(f"Dimensiones del dataset: {df.shape[0]} filas y {df.shape[1]} columnas."),
        # Más componentes que quieras agregar aquí
    ])