

import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import dash_bootstrap_components as dbc
from dash import html, dcc

def k_means_clustering_component(fullpath):
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
        dbc.CardHeader(html.H5("Minería de Datos")),
        dbc.CardBody([
            # Texto introductorio
            html.Div([
                html.H2("Segmentación de clientes con Clustering (K-Means)"),
                html.Blockquote([
                    html.H4("🎯 Objetivo de negocio:"),
                    html.P("Identificar los diferentes tipos de clientes que llegan al hotel, determinar si hay grupos que reservan con más antelación, y distinguir entre familias y viajeros de negocios.")
                ]),
                html.H4("💡 Idea"),
                html.P(
                    "Usar K-Means para agrupar clientes en perfiles en base a sus características:"
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
