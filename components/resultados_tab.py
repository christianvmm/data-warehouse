from dash import html
import dash_bootstrap_components as dbc
from utils.read_file import read_file

def resultados_tab(filepath):
    if not filepath:
        return dbc.Container([
            dbc.Alert("No hay datos cargados aún.", color="secondary", className="mt-4")
        ], fluid=True)

    try:
        df = read_file(filepath)
        return dbc.Container([
            dbc.Card(
                dbc.CardBody([
                    html.H5(
                        f"Tamaño del dataframe: {df.shape[0]} filas × {df.shape[1]} columnas",
                        className="text-primary mb-0"
                    )
                ]),
                className="shadow-sm rounded mt-4 p-3"
            )
        ], fluid=True, style={'maxWidth': '600px'})
    except Exception as e:
        return dbc.Container([
            dbc.Alert(f"Error leyendo archivo: {str(e)}", color="danger", className="mt-4")
        ], fluid=True)
