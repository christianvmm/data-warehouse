

import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import dash_bootstrap_components as dbc
from dash import html, dcc

# Variables globales
scaler = None
kmeans = None
descripciones = {
    0: "🔹 Cluster 0: Parejas o adultos solos, estadías cortas, pocas solicitudes.",
    1: "🔹 Cluster 1: Familias con niños, noches en fin de semana, prefieren habitaciones grandes.",
    2: "🔹 Cluster 2: Estancias largas, reservan con anticipación, posiblemente viajeros frecuentes.",
    3: "🔹 Cluster 3: Estadías más cortas, reservan en el último momento, buscan comodidad rápida."
}

def k_means_clustering_component(fullpath):
    global scaler, kmeans  # Agrega esto al principio de la función
    
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

    # --- FORMULARIO SIMPLIFICADO PARA NUEVA RESERVA ---
    form_inputs = html.Div([
        html.H6("Clasificar nueva reserva"),
        
        dbc.Row([
            dbc.Col([
                html.Label("Número de adultos"),
                dbc.Input(id='input_adults', type='number', value=2)
            ], md=3),
            dbc.Col([
                html.Label("Número de niños"),
                dbc.Input(id='input_children', type='number', value=4)
            ], md=3),
            dbc.Col([
                html.Label("Noches fin de semana"),
                dbc.Input(id='input_weekend', type='number', value=2)
            ], md=3),
            dbc.Col([
                html.Label("Noches entre semana"),
                dbc.Input(id='input_week', type='number', value=2)
            ], md=3),
        ], className='mb-2'),

        dbc.Row([
            dbc.Col([
                html.Label("Lead time (días)"),
                dbc.Input(id='input_lead_time', type='number', value=10)
            ], md=3),
            dbc.Col([
                html.Label("¿Estacionamiento? (0 o 1)"),
                dbc.Input(id='input_parking', type='number', value=0)
            ], md=3),
            dbc.Col([
                html.Label("Tipo habitación (codificado)"),
                dbc.Input(id='input_room', type='number', value=1)
            ], md=3),
            dbc.Col([
                html.Label("Segmento mercado (codificado)"),
                dbc.Input(id='input_market', type='number', value=1)
            ], md=3),
        ], className='mb-2'),

        dbc.Button("Clasificar reserva", id='btn_predict', color='primary', className='mt-2'),
        html.Div(id='prediction_result', className='mt-3')
    ])


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

            form_inputs
        ])
    ], className="mb-4 shadow-sm")




from dash import Input, Output, State, callback

@callback(
    Output('prediction_result', 'children'),
    Input('btn_predict', 'n_clicks'),
    State('input_adults', 'value'),
    State('input_children', 'value'),
    State('input_weekend', 'value'),
    State('input_week', 'value'),
    State('input_parking', 'value'),
    State('input_room', 'value'),
    State('input_lead_time', 'value'),
    State('input_market', 'value'),
)
def classify_new_reservation(n_clicks, adults, children, weekend, week, parking, room, lead_time, market):
    if not n_clicks:
        return ""
    try:
        # Validar entrada
        values = [adults, children, weekend, week, parking, room, lead_time, market]
        if any(v is None for v in values):
            return dbc.Alert("⚠️ Por favor completa todos los campos antes de predecir.", color="warning")

        # Crear DataFrame con una fila
        new_data = pd.DataFrame([values], columns=[
            'no_of_adults',
            'no_of_children',
            'no_of_weekend_nights',
            'no_of_week_nights',
            'required_car_parking_space',
            'room_type_reserved',
            'lead_time',
            'market_segment_type'
        ])

        # Escalar los datos
        new_scaled = scaler.transform(new_data)

        # Predecir con el modelo
        cluster = kmeans.predict(new_scaled)[0]
        descripcion = descripciones.get(cluster, "Segmento no identificado.")

        return dbc.Alert(f"📊 La reserva fue clasificada en el *Cluster {cluster}*. {descripcion}", color="info")
    except Exception as e:
        return dbc.Alert(f"❌ Error al predecir: {str(e)}", color="danger")
