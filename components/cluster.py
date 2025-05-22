from dash import callback, Output, Input, State, html, dcc
import dash
import pandas as pd
import os
import plotly.express as px
from sklearn.cluster import KMeans
import tempfile
from sklearn.preprocessing import StandardScaler

TMP_DIR = os.path.join(tempfile.gettempdir(), 'dash_uploads')

from dash import callback, Output, Input, State, html, dcc, dash
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import plotly.express as px

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

    if not numeric_cols:
        return html.Div("No hay columnas numéricas para clusterizar.")

    return html.Div([
        html.Label("Selecciona columnas predictoras (X):"),
        dcc.Dropdown(
            id='cluster-x-dropdown',
            options=[{'label': col, 'value': col} for col in numeric_cols],
            multi=True,
            value=numeric_cols[:2] if len(numeric_cols) >= 2 else numeric_cols
        ),
        html.Br(),
        html.Label("Selecciona número de clusters:"),
        dcc.Input(id='kmeans-n-clusters', type='number', value=3, min=1, max=10),
    ])



@callback(
    Output('mining-output-container', 'children', allow_duplicate=True),
    Input('cluster-x-dropdown', 'value'),
    Input('kmeans-n-clusters', 'value'),
    State('mining-technique-dropdown', 'value'),
    State('transformed-filepath', 'data'),
    prevent_initial_call = True
)
def aplicar_kmeans(x_cols, n_clusters, tecnica, processed_filename):
    if tecnica != 'kmeans' or not processed_filename or not x_cols or len(x_cols) < 1:
        raise dash.exceptions.PreventUpdate

    fullpath = os.path.join(TMP_DIR, processed_filename)
    df = pd.read_csv(fullpath)

    # Selección columnas para clustering
    X = df[x_cols].dropna().copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters or 3, random_state=0)
    kmeans.fit(X_scaled)

    X['cluster'] = kmeans.labels_.astype(str)

    # Solo para gráficar: si hay más de 2 columnas, hacemos PCA o usamos las dos primeras
    if len(x_cols) == 1:
        # Gráfico unidimensional
        fig = px.scatter(x=X_scaled[:, 0], y=[0]*len(X_scaled), color=X['cluster'],
                         labels={'x': x_cols[0], 'y': ''}, title='Clusters KMeans')
    elif len(x_cols) >= 2:
        fig = px.scatter(
            X, x=x_cols[0], y=x_cols[1], color='cluster',
            title=f'Clusters KMeans ({n_clusters} clusters)'
        )
        # Centroides en escala original (desescalamos centroides)
        centroids_scaled = kmeans.cluster_centers_
        centroids = scaler.inverse_transform(centroids_scaled)
        fig.add_scatter(x=centroids[:, 0], y=centroids[:, 1],
                        mode='markers',
                        marker=dict(size=15, color='black', symbol='x'),
                        name='Centroides')

    return dcc.Graph(figure=fig)
