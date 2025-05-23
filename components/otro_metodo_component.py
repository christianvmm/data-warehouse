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
                html.H2("Regresión para determinar el precio a cobrar"),
                html.Blockquote([
                    html.H4("🎯 Objetivo de negocio:"),
                    html.P(
                        "Predecir el precio promedio por habitación que pagará cada cliente según las características de su reserva, "
                        "permitiendo así optimizar la estrategia de precios y maximizar los ingresos."
                    )
                ]),
                html.H4("💡 Idea"),
                html.P(
                    "Utilizar un modelo de regresión que, a partir de datos históricos de reservas, "
                    "aprenda patrones relacionados con el precio que los clientes tienden a pagar bajo diferentes condiciones."
                ),
                html.H4("✅ ¿Por qué es útil para el negocio?"),
                html.Ul([
                    html.Li("Permite anticipar ingresos por reserva antes de la confirmación, ayudando a planificar recursos y promociones."),
                    html.Li("Facilita la implementación de estrategias de precios dinámicos basados en el perfil y comportamiento del cliente."),
                    html.Li("Mejora la toma de decisiones sobre descuentos o paquetes especiales, ajustándolos a la probabilidad de pago real."),
                    html.Li("Ayuda a identificar segmentos de clientes que pagan más, para orientar campañas de marketing específicas."),
                ]),
                html.Hr()
            ], style={'marginBottom': '30px'}),
        ])
    ], className="mb-4 shadow-sm")

