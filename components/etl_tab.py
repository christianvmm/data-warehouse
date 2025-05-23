from dash import html, dcc
import dash_bootstrap_components as dbc
from utils.file_to_df import file_to_df

def etl_tab(filepath):
    if filepath is None:
        return html.Div("No se ha subido ningún archivo.")

    df = file_to_df(filepath)

    return dbc.Container(
        fluid=True,
        style={"backgroundColor": "#f2f2f2", "minHeight": "100vh", "padding": "30px"},
        children=[
            html.H2("Procesamiento ETL", className="text-center mb-4"),

           
            dbc.Row([

                # LIMPIEZA
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody([
                            html.H4("1. Limpieza de Datos", className="mb-3"),
                            html.Label("Eliminar columnas:"),
                            dcc.Dropdown(
                                id='etl-drop-columns',
                                options=[{'label': col, 'value': col} for col in df.columns],
                                multi=True,
                                placeholder='Selecciona columnas a eliminar...'
                            ),
                            html.Br(),
                            html.Label("Convertir columnas a fecha:"),
                            dcc.Dropdown(
                                id='etl-convert-date',
                                options=[{'label': col, 'value': col} for col in df.columns],
                                multi=True,
                                placeholder='Selecciona columnas de fecha...'
                            ),
                            html.Br(),
                            html.Label("Opciones adicionales:"),
                            dbc.Checklist(
                                id='etl-options',
                                options=[
                                    {'label': 'Reemplazar valores nulos', 'value': 'nulls'},
                                ],
                                value=[],
                                switch=True
                            )
                        ]),
                        className="shadow-sm rounded-4 p-4 bg-white",
                    ),
                    width=6
                ),

               
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody([
                            html.H4("2. Transformación de Datos", className="mb-3"),
                            html.Label("Columnas a normalizar:"),
                            dcc.Dropdown(
                                id='etl-normalize-columns',
                                options=[{'label': col, 'value': col} for col in df.select_dtypes(include='number').columns],
                                multi=True,
                                placeholder='Selecciona columnas numéricas...'
                            ),
                            html.Br(),
                            html.Label("Filtro por valores numéricos:"),
                            dcc.Dropdown(
                                id='etl-filter-column',
                                options=[{'label': col, 'value': col} for col in df.select_dtypes(include='number').columns],
                                placeholder='Columna numérica...'
                            ),
                            # html.Br(),
                            # html.Label("Columnas a convertir a formato numérico:"),
                            # dcc.Dropdown(
                            #     id='etl-to-numeric-columns',
                            #     options=[{'label': col, 'value': col} for col in df.select_dtypes(exclude='number').columns],
                            #     multi=True,
                            #     placeholder='Selecciona columnas no numéricas...'
                            # ),
                            # dbc.Row([
                            #     dbc.Col(dcc.Input(id='etl-filter-min', type='number', placeholder='Mínimo'), width=6),
                            #     dbc.Col(dcc.Input(id='etl-filter-max', type='number', placeholder='Máximo'), width=6),
                            # ], className="mt-2")
                        ]),
                        className="shadow-sm rounded-4 p-4 bg-white",
                    ),
                    width=6
                )
            ], className="mb-4"),

            
            dbc.Row(
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody([

                            
                            dbc.Button("Aplicar transformaciones", id='etl-apply-button', n_clicks=0, color="primary", className="mb-3"),
                            html.Div(id='etl-feedback', style={'color': 'green', 'marginBottom': '20px'}),

                            
                            html.H5("Vista previa del DataFrame transformado", className="mb-3"),
                            html.Div(id='etl-table'),

                            html.Hr(),

                            
                            # html.H5("Descargar datos procesados", className="mt-4"),
                            # dbc.Row([
                            #     dbc.Col([
                            #         html.Label("Formato de descarga:"),
                            #         dbc.RadioItems(
                            #             id='download-format',
                            #             options=[
                            #                 {'label': 'CSV', 'value': 'csv'},
                            #                 {'label': 'JSON', 'value': 'json'},
                            #                 {'label': 'Excel', 'value': 'excel'}
                            #             ],
                            #             value='csv',
                            #             inline=True
                            #         )
                            #     ], width=8),
                            #     dbc.Col([
                            #         dbc.Button("Descargar archivo", id='download-button', n_clicks=0, color="success", className="mt-3")
                            #     ], width=4)
                            # ])
                        ]),
                        className="shadow-sm rounded-4 p-4 bg-white",
                    ),
                    width=12
                )
            ),

            
            dcc.Store(id='transformed-filepath', storage_type='session'),
            dcc.Download(id="download-data")
        ]
    )