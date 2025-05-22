# from dash import html
# from utils.read_file import read_file
# from dash import dash_table


# def upload_tab(filepath):
#     children = []

#     if filepath:
#         try:
#             df = read_file(filepath)
#             children.append(render_table(df))
#         except Exception as e:
#             children.append(html.Div(f"Error leyendo archivo: {str(e)}"))

#     return children


# def render_table(df):
#     return dash_table.DataTable(
#         df.to_dict('records'),
#         [{'name': i, 'id': i} for i in df.columns],
#         style_table={'overflowX': 'auto'},
#         page_size=10
#     )
from dash import html, dash_table
import dash_bootstrap_components as dbc
from utils.file_to_df import file_to_df

def upload_tab(filepath):
    if not filepath:
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    dbc.Alert(
                        "No se ha subido ningún archivo.",
                        color="warning",
                        style={
                            "backgroundColor": "#fff8e1", 
                            "borderColor": "#ffeeba",
                            "color": "#856404"
                        },
                        dismissable=True
                    )
                ])
            ])
        ])

    try:
        df = file_to_df(filepath)
    except Exception as e:
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    dbc.Alert(
                        f"Error al leer el archivo: {str(e)}",
                        color="danger",
                        style={
                            "backgroundColor": "#f8d7da",
                            "borderColor": "#f5c6cb",
                            "color": "#721c24"
                        },
                        dismissable=True
                    )
                ])
            ])
        ])

    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H5(
                    "📄 Vista previa del archivo",
                    className="text-center mb-4",
                    style={"color": "#1a237e"} 
                )
            ])
        ]),
        dbc.Row([
            dbc.Col([
                dbc.Card(
                    dbc.CardBody([
                        dash_table.DataTable(
                            data=df.head(10).to_dict('records'),
                            columns=[{'name': col, 'id': col} for col in df.columns],
                            page_size=10,
                            style_table={'overflowX': 'auto'},
                            style_header={
                                'backgroundColor': "#3f5cad",
                                'color': 'white',
                                'fontWeight': 'bold'
                            },
                            style_cell={'padding': '10px', 'textAlign': 'left'}
                        )
                    ]),
                    className="shadow-sm rounded"
                )
            ], width=12)
        ])
    ], fluid=True)