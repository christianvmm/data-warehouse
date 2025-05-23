import os
import tempfile
import pandas as pd
from dash import html, dcc, dash_table
import dash_bootstrap_components as dbc

TMP_DIR = os.path.join(tempfile.gettempdir(), 'dash_uploads')

def mineria_tab(processed_filename):
    if processed_filename is None:
        return dbc.Container([
            dbc.Alert("No hay datos procesados desde ETL.", color="secondary", className="mt-4")
        ], fluid=True)

    fullpath = os.path.join(TMP_DIR, processed_filename)
    if not os.path.exists(fullpath):
        return dbc.Container([
            dbc.Alert("Archivo procesado no encontrado.", color="danger", className="mt-4")
        ], fluid=True)

    df = pd.read_csv(fullpath)
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    if not numeric_cols:
        return dbc.Container([
            dbc.Alert("No se encontraron columnas numéricas.", color="warning", className="mt-4")
        ], fluid=True)

    
    numeric_df = df[numeric_cols]
    stats_df = numeric_df.describe().T
    stats_df["Mediana"] = numeric_df.median()
    stats_df.rename(columns={
        'count': 'Cantidad',
        'mean': 'Media',
        'std': 'Desviación Estándar',
        'min': 'Mínimo',
        'max': 'Máximo'
    }, inplace=True)
    stats_df.reset_index(inplace=True)
    stats_df.rename(columns={'index': 'Variable'}, inplace=True)

    stats_table = dash_table.DataTable(
        data=stats_df.to_dict('records'),
        columns=[{"name": col, "id": col} for col in stats_df.columns],
        page_size=10,
        style_table={'overflowX': 'auto'},
        style_header={'backgroundColor': '#1e3a8a', 'color': 'white'},
        style_cell={'padding': '5px', 'textAlign': 'left'}
    )

    return dbc.Container(fluid=True, style={'paddingTop': '30px', 'paddingBottom': '30px'}, children=[
        dbc.Row(dbc.Col(html.H4("Exploración y Minería de Datos", className="text-center text-primary mb-4"))),

        dbc.Row([
            dbc.Col(html.P(f"Dimensiones del dataset: {df.shape[0]} filas × {df.shape[1]} columnas."),
                    width=12, className="mb-4")
        ]),

        
        dbc.Card([
            dbc.CardHeader(html.H5("Estadísticas Descriptivas de Columnas Numéricas")),
            dbc.CardBody(stats_table)
        ], className="mb-4 shadow-sm"),

        
        dbc.Card([
            dbc.CardHeader(html.H5("Visualización de Datos Numéricos")),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.Label("Selecciona una columna numérica:", className="form-label"),
                        dcc.Dropdown(
                            id='eda-numeric-dropdown',
                            options=[{'label': col, 'value': col} for col in numeric_cols],
                            value=numeric_cols[0],
                            clearable=False
                        )
                    ], width=6)
                ], className="mb-3"),
                html.Div(id='eda-plots-container')
            ])
        ], className="mb-4 shadow-sm"),

        
        kmeans_clustering_component(fullpath),
        
        dcc.Store(id='transformed-filepath', data=processed_filename)
    ])



import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import dash_bootstrap_components as dbc
from dash import html, dcc

def kmeans_clustering_component(fullpath):
    # Carga datos
    df = pd.read_csv(fullpath)

    # Variables para segmentar
    cols = [
        'no_of_adults',
        'no_of_children',
        'no_of_weekend_nights',
        'no_of_week_nights',
        'required_car_parking_space',
        'room_type_reserved',
        'lead_time',
        'market_segment_type'
    ]

    # Label Encoding para variables categóricas
    le_room = LabelEncoder()
    le_market = LabelEncoder()

    df['room_type_reserved'] = le_room.fit_transform(df['room_type_reserved'])
    df['market_segment_type'] = le_market.fit_transform(df['market_segment_type'])

    # Escalado
    X = df[cols]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Método del codo
    sse = []
    for k in range(1, 10):
        kmeans_test = KMeans(n_clusters=k, random_state=42)
        kmeans_test.fit(X_scaled)
        sse.append(kmeans_test.inertia_)

    # Gráfico método del codo con Plotly
    elbow_fig = go.Figure()
    elbow_fig.add_trace(go.Scatter(x=list(range(1, 10)), y=sse, mode='lines+markers'))
    elbow_fig.update_layout(
        title='Método del Codo',
        xaxis_title='Número de Clusters',
        yaxis_title='SSE',
        template='plotly_white'
    )

    # KMeans final con k=4
    kmeans = KMeans(n_clusters=4, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    df['cluster'] = clusters

    # PCA para visualización
    pca = PCA(n_components=2)
    components = pca.fit_transform(X_scaled)
    df['pca1'] = components[:, 0]
    df['pca2'] = components[:, 1]

    # Gráfico PCA con clusters
    pca_fig = px.scatter(
        df, x='pca1', y='pca2', color='cluster',
        title='Segmentación de Clientes (Clustering)',
        color_discrete_sequence=px.colors.qualitative.T10,  # Usar secuencia discreta para clusters
        template='plotly_white',
        labels={'pca1': 'PCA 1', 'pca2': 'PCA 2'}
    )

    # Resumen por cluster
    cluster_summary = df.groupby('cluster')[cols].mean().reset_index()

    # Crear tabla resumen con Dash
    table_header = [
        html.Thead(html.Tr([html.Th(col) for col in ['cluster'] + cols]))
    ]
    rows = []
    for _, row in cluster_summary.iterrows():
        rows.append(html.Tr([html.Td(row['cluster'])] + [html.Td(f"{row[c]:.2f}") for c in cols]))
    table_body = [html.Tbody(rows)]
    summary_table = dbc.Table(table_header + table_body, bordered=True, hover=True, responsive=True, striped=True)

    # Descripciones manuales
    descripciones = {
        0: "🔹 Cluster 0: Parejas o adultos solos, estadías cortas, pocas solicitudes.",
        1: "🔹 Cluster 1: Familias con niños, noches en fin de semana, prefieren habitaciones grandes.",
        2: "🔹 Cluster 2: Estancias largas, reservan con anticipación, posiblemente viajeros frecuentes.",
        3: "🔹 Cluster 3: Estadías más cortas, reservan en el último momento, buscan comodidad rápida."
    }
    desc_items = [html.Li(desc) for desc in descripciones.values()]

    # Layout final con todos los componentes
    return dbc.Card([
        dbc.CardHeader(html.H5("Minería de Datos - Clustering KMeans")),
        dbc.CardBody([
            # Texto introductorio
            html.Div([
                html.H2("🎯 Caso de uso: Segmentación de clientes con Clustering (K-Means)"),
                html.Blockquote([
                    html.H4("❓ Pregunta de negocio:"),
                    html.P("¿Qué tipos de clientes diferentes llegan al hotel? ¿Hay grupos que reserven con más antelación? ¿Familias? ¿Viajeros de negocios?")
                ]),
                html.H4("💡 Idea"),
                html.P(
                    "Usamos K-Means Clustering para agrupar clientes en perfiles en base a sus características:"
                ),
                html.Ul([
                    html.Li("¿Cuántos adultos y niños traen?"),
                    html.Li("¿Cuánto tiempo se quedan?"),
                    html.Li("¿Reservan con cuánta anticipación?"),
                    html.Li("¿Qué tipo de habitación eligen?"),
                    html.Li("¿Piden estacionamiento o no?"),
                    html.Li("¿Vienen por qué canal?")
                ]),
                html.H4("✅ ¿Por qué es útil para el negocio?"),
                html.Ul([
                    html.Li("Permite crear campañas personalizadas por segmento."),
                    html.Li("El hotel puede optimizar precios, servicios y paquetes para cada grupo."),
                    html.Li("Se pueden identificar clientes VIP, viajeros frecuentes, familias, etc.")
                ]),
                html.Hr()
            ], style={'marginBottom': '30px'}),


            html.H6("Método del Codo"),
            dcc.Graph(figure=elbow_fig),

            html.H6("Visualización PCA"),
            dcc.Graph(figure=pca_fig),

            html.H6("Resumen promedio por cluster"),
            summary_table,

            html.H6("Descripción general de los clusters"),
            html.Ul(desc_items),
        ])
    ], className="mb-4 shadow-sm")
