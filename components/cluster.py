from dash import callback, Output, Input, State, html, dcc
import dash
import pandas as pd
import os
import plotly.express as px
from sklearn.cluster import KMeans
import tempfile
TMP_DIR = os.path.join(tempfile.gettempdir(), 'dash_uploads')

@callback(
    Output('cluster-variable-selectors', 'children'),
    Input('mining-technique-dropdown', 'value'),
    State('transformed-filepath', 'data'),
)
def mostrar_dropdowns_cluster(tecnica, processed_filename):
    if tecnica != 'kmeans' or not processed_filename:
        return []

    fullpath = os.path.join(TMP_DIR, processed_filename)
    df = pd.read_csv(fullpath)
    numeric_cols = df.select_dtypes(include='number').columns.tolist()

    options = [{'label': col, 'value': col} for col in numeric_cols]
    return html.Div([
        html.Label("Columna X:"),
        dcc.Dropdown(id='cluster-x-dropdown', options=options, value=numeric_cols[0]),
        html.Label("Columna Y:"),
        dcc.Dropdown(id='cluster-y-dropdown', options=options, value=numeric_cols[1])
    ])

# @callback(
#     Output('mining-output-container', 'children'),
#     Input('cluster-x-dropdown', 'value'),
#     Input('cluster-y-dropdown', 'value'),
#     State('mining-technique-dropdown', 'value'),
#     State('transformed-filepath', 'data'),
# )
def aplicar_kmeans(x_col, y_col, tecnica, processed_filename):
    if tecnica != 'kmeans':
        raise dash.exceptions.PreventUpdate

    fullpath = os.path.join(TMP_DIR, processed_filename)
    df = pd.read_csv(fullpath)
    X = df[[x_col, y_col]].dropna()
    kmeans = KMeans(n_clusters=3).fit(X)
    X['cluster'] = kmeans.labels_

    fig = px.scatter(X, x=x_col, y=y_col, color=X['cluster'].astype(str))
    return dcc.Graph(figure=fig)