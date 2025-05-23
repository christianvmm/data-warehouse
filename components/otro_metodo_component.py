import pandas as pd
import numpy as np
from dash import html, dcc, dash_table
import dash_bootstrap_components as dbc
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px

def otro_metodo_component(fullpath):
    # Carga datos
    df = pd.read_csv(fullpath)

    # LIMPIAR nombres columnas para evitar problemas con espacios
    df.columns = df.columns.str.strip()

    # LIMPIEZA BÁSICA
    df.dropna(inplace=True)

    # CODIFICAR VARIABLES CATEGÓRICAS
    label_encoders = {}
    for col in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Variable objetivo y preparación de datos
    drop_cols = ['avg_price_per_room']
    if 'booking_status' in df.columns:
        drop_cols.append('booking_status')

    X = df.drop(columns=drop_cols)
    y = df['avg_price_per_room']

    # División train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Entrenamiento
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predicción y métricas
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Importancia variables
    importances = model.feature_importances_
    feature_names = X.columns
    indices = np.argsort(importances)[::-1]

    importance_table = pd.DataFrame({
        'Feature': feature_names[indices],
        'Importance': importances[indices]
    })

    # Gráfica de importancia con Plotly Express
    fig = px.bar(
        importance_table,
        x='Feature',
        y='Importance',
        title='Importancia de variables para predecir avg_price_per_room',
        labels={'Importance': 'Importancia', 'Feature': 'Variable'},
        text=importance_table['Importance'].apply(lambda x: f"{x:.3f}")
    )
    fig.update_traces(textposition='outside')
    fig.update_layout(margin=dict(t=50, b=50, l=50, r=50), height=900, xaxis_tickangle=-45)

    return dbc.Card([
        dbc.CardHeader(html.H5("Minería de Datos")),
        dbc.CardBody([
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
                html.Hr(),

                # Métricas
                html.H4("📊 Resultados del modelo"),
                html.P(f"Mean Squared Error (MSE): {mse:.2f}"),
                html.P(f"R² Score: {r2:.2f}"),

                # Tabla importancia variables
                html.H5("Importancia de variables"),
                # Gráfica Plotly
                html.H5("Gráfica de importancia de variables"),
                dcc.Graph(figure=fig)
            ], style={'marginBottom': '30px'}),
        ])
    ], className="mb-4 shadow-sm")
