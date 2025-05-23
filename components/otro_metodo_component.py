import pandas as pd
from dash import html, dcc, dash_table
import dash_bootstrap_components as dbc

def otro_metodo_component(fullpath):
    # Carga datos
    df = pd.read_csv(fullpath)

    return dbc.Card([
        dbc.CardHeader(html.H5("Minería de Datos")),
        dbc.CardBody([
            # Texto introductorio
            html.Div([
                html.H2("Otro método de weba"),
                html.Blockquote([
                    html.H4("🎯 Objetivo de negocio:"),
                    html.P("Lorem")
                ]),
                html.H4("💡 Idea"),
                html.P(
                    "Lorem"
                ),
               
                html.H4("✅ ¿Por qué es útil para el negocio?"),
                html.Ul([
                    html.Li("X"),
                    html.Li("Y"),
                ]),
                html.Hr()
            ], style={'marginBottom': '30px'}),
        ])
    ], className="mb-4 shadow-sm")