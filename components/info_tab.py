# from dash import  html
# from utils.read_file import read_file

# def info_tab(filepath):
#   if not filepath:
#       return html.Div("No hay datos cargados aún.")


#   try:
#       df = read_file(filepath)
#       return html.Div([
#           html.H5(f"Tamaño del DataFrame: {df.shape[0]} filas x {df.shape[1]} columnas")
#       ])
#   except Exception as e:
#       return html.Div(f"Error leyendo archivo: {str(e)}")


# components/info_tab.py
from dash import html
import dash_bootstrap_components as dbc
from utils.file_to_df import file_to_df
import math

def info_tab(filepath):
    if not filepath:
        return dbc.Container([
            dbc.Alert(
                "No se ha subido ningún archivo.",
                color="secondary",
                className="mt-4"
            )
        ], fluid=True)

    try:
        df = file_to_df(filepath)
    except Exception as e:
        return dbc.Container([
            dbc.Alert(
                f"Error al cargar archivo: {str(e)}",
                color="danger",
                className="mt-4"
            )
        ], fluid=True)

    
    col_data = [f"{col} ({dtype})" for col, dtype in zip(df.columns, df.dtypes)]
    n = len(col_data)
    chunk = math.ceil(n / 2)
    left_cols = col_data[:chunk]
    right_cols = col_data[chunk:]

    
    nulos = df.isnull().sum()
    nul_list = [f"{col}: {nulos[col]} nulos" for col in df.columns if nulos[col] > 0]

    return dbc.Container([
        
        html.H4("Información general del archivo",
                className="text-primary text-center mb-4 mt-4"),

        
        dbc.Row([
            dbc.Col(html.P(f"Filas: {df.shape[0]}", className="h5 text-center"), width=6),
            dbc.Col(html.P(f"Columnas: {df.shape[1]}", className="h5 text-center"), width=6)
        ], className="mb-4"),

        
        html.H5("Columnas y tipos de datos:", className="text-info mb-3"),
        dbc.Row([
            dbc.Col(
                html.Ul([html.Li(item, className="list-group-item") for item in left_cols],
                        className="list-group"),
                width=6
            ),
            dbc.Col(
                html.Ul([html.Li(item, className="list-group-item") for item in right_cols],
                        className="list-group"),
                width=6
            )
        ], className="mb-4"),

        
        html.H5("Valores nulos:", className="text-info mb-3"),
        dbc.Row([
            dbc.Col(
                html.Ul(
                    [html.Li(item, className="list-group-item") for item in nul_list]
                    if nul_list else [html.Li("No hay valores nulos.", className="list-group-item text-muted")],
                    className="list-group"
                ),
                width=12
            )
        ]),

    ], fluid=True, style={'maxWidth': '900px'})
