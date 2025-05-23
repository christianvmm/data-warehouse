import dash_bootstrap_components as dbc
from dash import html

def resultados_tab(filepath):
    return dbc.Container([
        # Conclusiones del modelo de clustering (K-Means)
        dbc.Card([
            dbc.CardHeader(html.H4("📌 Conclusión y Toma de Decisión", className="card-title")),
            dbc.CardBody([
                html.P(
                    "Este proyecto demuestra cómo la minería de datos, aplicada mediante el algoritmo K-Means, permite segmentar eficientemente a los clientes de un hotel en distintos perfiles basados en patrones de comportamiento detectados en las reservaciones históricas.",
                    style={"fontSize": "1.1rem"}
                ),
                html.H5("🔍 Principales hallazgos", className="mt-4"),
                html.Ul([
                    html.Li("El modelo clasifica nuevas reservaciones automáticamente en uno de los clusters definidos."),
                    html.Li("Cada cluster representa un perfil de cliente con necesidades y comportamientos similares."),
                    html.Li("El modelo es altamente útil para la personalización de estrategias comerciales."),
                ], style={"fontSize": "1rem"}),

                html.H5("🚀 Aplicación práctica en el negocio", className="mt-4"),
                html.P(
                    "Implementar este algoritmo en el sistema de reservas del hotel permitiría automatizar la clasificación de cada nuevo cliente en tiempo real. Inmediatamente después de una reservación, el sistema podría activar acciones automatizadas, como:",
                    style={"fontSize": "1.05rem"}
                ),
                html.Ul([
                    html.Li("Enviar un correo electrónico con promociones exclusivas según su perfil."),
                    html.Li("Recomendar upgrades de habitación o servicios adicionales personalizados."),
                    html.Li("Incluir al cliente en campañas de fidelización o beneficios especiales."),
                ], style={"fontSize": "1rem"}),

                html.P(
                    "Por ejemplo, si una familia reserva con anticipación, el sistema podría enviarle un correo sugiriendo un paquete familiar con desayuno incluido. O si se detecta un viajero frecuente de negocios, podría ofrecerle un check-in express o descuentos por estadías repetidas.",
                    style={"marginTop": "10px", "fontSize": "1.05rem"}
                ),

                html.Hr(),

                html.P(
                    "👉 En resumen: el modelo no solo analiza datos históricos, sino que se convierte en una herramienta predictiva capaz de integrarse con el sistema de reservas para activar comunicaciones automatizadas y personalizadas. Esto representa una clara ventaja competitiva para el hotel.",
                    style={"fontSize": "1.1rem", "fontWeight": "bold", "color": "#0d6efd"}
                )
            ])
        ], className="shadow-sm p-4 mt-3"),

        # Conclusiones del modelo de regresión
        dbc.Card([
            dbc.CardHeader(html.H4("📈 Conclusiones del modelo de Regresión", className="card-title")),
            dbc.CardBody([
                html.P(
                    "Además del análisis de segmentación, se implementó un modelo de regresión para predecir el precio promedio por habitación que pagará cada cliente en función de las características de su reserva.",
                    style={"fontSize": "1.1rem"}
                ),
                html.H5("🔍 Principales hallazgos del modelo", className="mt-4"),
                html.Ul([
                    html.Li("El modelo utiliza algoritmos de regresión para predecir el precio con una buena precisión."),
                    html.Li("Las variables con mayor impacto en la predicción incluyen el tipo de habitación, el mes de llegada, y el segmento de mercado."),
                    html.Li("El modelo puede integrarse fácilmente con el sistema de reservas para calcular automáticamente un precio sugerido personalizado."),
                ], style={"fontSize": "1rem"}),

                html.H5("💼 Aplicación directa en el negocio", className="mt-4"),
                html.P(
                    "Este modelo permite establecer precios dinámicos por reserva, basados en datos reales y ajustados al perfil del cliente. Esto aporta múltiples beneficios estratégicos:",
                    style={"fontSize": "1.05rem"}
                ),
                html.Ul([
                    html.Li("Maximiza los ingresos ajustando el precio según la demanda y características del cliente."),
                    html.Li("Facilita la implementación de descuentos estratégicos o promociones automáticas."),
                    html.Li("Permite realizar simulaciones para evaluar el impacto de campañas o políticas de precios."),
                ], style={"fontSize": "1rem"}),

                html.Hr(),

                html.P(
                    "✅ En resumen: este modelo permite transformar el sistema de precios del hotel en un proceso inteligente y automatizado, mejorando la rentabilidad y ofreciendo una experiencia más personalizada al cliente.",
                    style={"fontSize": "1.1rem", "fontWeight": "bold", "color": "#198754"}
                )
            ])
        ], className="shadow-sm p-4 mt-3")
    ], fluid=True)
