from dash import  html
from utils.read_file import read_file

def mineria_tab(filepath):
    if not filepath:
        return html.Div("No hay datos cargados aún.")
    
    try:
        df = read_file(filepath)
        return html.Div([
            html.H5(f"Tamaño del DataFrame: {df.shape[0]} filas x {df.shape[1]} columnas")
        ]) 
    except Exception as e:
        return html.Div(f"Error leyendo archivo: {str(e)}")