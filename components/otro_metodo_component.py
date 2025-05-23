import pandas as pd
import numpy as np
from dash import html, dcc, dash_table, Input, Output, State, callback
import dash_bootstrap_components as dbc
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px

# Variables globales
state = None

def otro_metodo_component(fullpath):
    global state 

    # --- Carga y preprocesamiento ---
    df = pd.read_csv(fullpath)
    df.columns = df.columns.str.strip()
    df.dropna(inplace=True)

    label_encoders = {}
    for col in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    drop_cols = ['avg_price_per_room']
    if 'booking_status' in df.columns:
        drop_cols.append('booking_status')

    X = df.drop(columns=drop_cols)
    y = df['avg_price_per_room']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    importances = model.feature_importances_
    feature_names = X.columns
    indices = np.argsort(importances)[::-1]

    importance_table = pd.DataFrame({
        'Feature': feature_names[indices],
        'Importance': importances[indices]
    })

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

    # Guardamos label_encoders y model en un diccionario para acceder en callback
    # Nota: en una app real mejor usar dcc.Store o servidor de backend
    # Aquí por simplicidad los guardamos en variables de cierre del callback
    state = {
        'label_encoders': label_encoders,
        'model': model,
        'feature_order': list(feature_names)
    }

    # Construcción UI
    card = dbc.Card([
        dbc.CardHeader(html.H5("Minería de Datos")),
        dbc.CardBody([
            html.Div([
                html.H2("Regresión para determinar el precio a cobrar"),
                html.Blockquote([
                    html.H4("Objetivo de negocio:"),
                    html.P(
                        "Predecir el precio promedio por habitación que pagará cada cliente según las características de su reserva, "
                        "permitiendo así optimizar la estrategia de precios y maximizar los ingresos."
                    )
                ]),
                html.H4("Idea"),
                html.P(
                    "Utilizar un modelo de regresión que, a partir de datos históricos de reservas, "
                    "aprenda patrones relacionados con el precio que los clientes tienden a pagar bajo diferentes condiciones."
                ),
                html.H4("¿Por qué es útil para el negocio?"),
                html.Ul([
                    html.Li("Permite anticipar ingresos por reserva antes de la confirmación, ayudando a planificar recursos y promociones."),
                    html.Li("Facilita la implementación de estrategias de precios dinámicos basados en el perfil y comportamiento del cliente."),
                    html.Li("Mejora la toma de decisiones sobre descuentos o paquetes especiales, ajustándolos a la probabilidad de pago real."),
                    html.Li("Ayuda a identificar segmentos de clientes que pagan más, para orientar campañas de marketing específicas."),
                ]),
                html.Hr(),

                # Métricas
                html.H4("Resultados del modelo"),
                html.P(f"Mean Squared Error (MSE): {mse:.2f}"),
                html.P(f"R² Score: {r2:.2f}"),

                # Tabla importancia variables
                html.H5("Importancia de variables"),

                # Gráfica Plotly
                dcc.Graph(figure=fig),

                html.Hr(),

                # Formulario para predicción nueva reserva
                html.H4("Predecir precio para nueva reserva"),

                # Inputs para las features (solo un subset para ejemplo, ajusta según tus columnas)
                dbc.Row([
                    dbc.Col([
                        dbc.Label("No. de adultos"),
                        dbc.Input(id='input_no_of_adults', type='number', min=0, step=1, value=2)
                    ], width=3),
                    dbc.Col([
                        dbc.Label("No. de niños"),
                        dbc.Input(id='input_no_of_children', type='number', min=0, step=1, value=3)
                    ], width=3),
                    dbc.Col([
                        dbc.Label("No. noches fin de semana"),
                        dbc.Input(id='input_no_of_weekend_nights', type='number', min=0, step=1, value=2)
                    ], width=3),
                    dbc.Col([
                        dbc.Label("No. noches entre semana"),
                        dbc.Input(id='input_no_of_week_nights', type='number', min=0, step=1, value=2)
                    ], width=3),
                ], className='mb-3'),

                dbc.Row([
                    dbc.Col([
                        dbc.Label("Tipo plan de comida"),
                        dcc.Dropdown(
                            id='input_type_of_meal_plan',
                            options=[{'label': val, 'value': idx} for val, idx in 
                                zip(label_encoders['type_of_meal_plan'].classes_, 
                                    label_encoders['type_of_meal_plan'].transform(label_encoders['type_of_meal_plan'].classes_))],
                            value=label_encoders['type_of_meal_plan'].transform([label_encoders['type_of_meal_plan'].classes_[0]])[0]
                        )
                    ], width=3),
                    dbc.Col([
                        dbc.Label("Espacio para auto"),
                        dcc.Dropdown(
                            id='input_required_car_parking_space',
                            options=[{'label': str(i), 'value': i} for i in [0,1]],
                            value=0
                        )
                    ], width=3),
                    dbc.Col([
                        dbc.Label("Tipo de habitación", className='text-danger'),
                        dcc.Dropdown(
                            id='input_room_type_reserved',
                            options=[{'label': val, 'value': idx} for val, idx in 
                                zip(label_encoders['room_type_reserved'].classes_, 
                                    label_encoders['room_type_reserved'].transform(label_encoders['room_type_reserved'].classes_))],
                            value=label_encoders['room_type_reserved'].transform([label_encoders['room_type_reserved'].classes_[0]])[0],
                        )
                    ], width=3),
                    dbc.Col([
                        dbc.Label("Lead time (días)"),
                        dbc.Input(id='input_lead_time', type='number', min=0, step=1, value=30)
                    ], width=3),
                ], className='mb-3'),

                dbc.Row([
                    dbc.Col([
                        dbc.Label("Año llegada"),
                        dbc.Input(id='input_arrival_year', type='number', min=2020, step=1, value=2025)
                    ], width=3),
                    dbc.Col([
                        dbc.Label("Mes llegada", className='text-danger'),
                        dbc.Input(id='input_arrival_month', type='number', min=1, max=12, step=1, value=9)
                    ], width=3),
                    dbc.Col([
                        dbc.Label("Día llegada"),
                        dbc.Input(id='input_arrival_date', type='number', min=1, max=31, step=1, value=15)
                    ], width=3),
                    dbc.Col([
                        dbc.Label("Segmento mercado", className='text-danger'),
                        dcc.Dropdown(
                            id='input_market_segment_type',
                            options=[{'label': val, 'value': idx} for val, idx in 
                                zip(label_encoders['market_segment_type'].classes_, 
                                    label_encoders['market_segment_type'].transform(label_encoders['market_segment_type'].classes_))],
                            value=label_encoders['market_segment_type'].transform([label_encoders['market_segment_type'].classes_[0]])[0],
                        )
                    ], width=3),
                ], className='mb-3'),

                dbc.Row([
                    dbc.Col([
                        dbc.Label("Huésped repetido"),
                        dcc.Dropdown(
                            id='input_repeated_guest',
                            options=[{'label': str(i), 'value': i} for i in [0,1]],
                            value=0
                        )
                    ], width=3),
                    dbc.Col([
                        dbc.Label("Prev cancelaciones"),
                        dbc.Input(id='input_no_of_previous_cancellations', type='number', min=0, step=1, value=0)
                    ], width=3),
                    dbc.Col([
                        dbc.Label("Prev reservas no canceladas"),
                        dbc.Input(id='input_no_of_previous_bookings_not_canceled', type='number', min=0, step=1, value=0)
                    ], width=3),
                    dbc.Col([
                        dbc.Label("No. de solicitudes especiales"),
                        dbc.Input(id='input_no_of_special_requests', type='number', min=0, step=1, value=1)
                    ], width=3),
                ], className='mb-3'),

                dbc.Button("Predecir Precio", id='btn_predict_regression', color='primary'),

                html.Hr(),
                html.Div(id='prediction_output', style={'marginTop': '20px', 'fontWeight': 'bold', 'fontSize': '1.2em'}),
            ], style={'marginBottom': '30px'}),
        ])
    ], className="mb-4 shadow-sm")

    

    return card



# CALLBACK dentro de la función, usando closure para acceso a modelo y label_encoders
@callback(
    Output('prediction_output', 'children'),
    Input('btn_predict_regression', 'n_clicks'),
    State('input_no_of_adults', 'value'),
    State('input_no_of_children', 'value'),
    State('input_no_of_weekend_nights', 'value'),
    State('input_no_of_week_nights', 'value'),
    State('input_type_of_meal_plan', 'value'),
    State('input_required_car_parking_space', 'value'),
    State('input_room_type_reserved', 'value'),
    State('input_lead_time', 'value'),
    State('input_arrival_year', 'value'),
    State('input_arrival_month', 'value'),
    State('input_arrival_date', 'value'),
    State('input_market_segment_type', 'value'),
    State('input_repeated_guest', 'value'),
    State('input_no_of_previous_cancellations', 'value'),
    State('input_no_of_previous_bookings_not_canceled', 'value'),
    State('input_no_of_special_requests', 'value'),
)
def predict_price(n_clicks, adults, children, weekend_nights, week_nights, meal_plan, car_parking,
                  room_type, lead_time, year, month, date, market_segment,
                  repeated_guest, prev_cancel, prev_not_cancel, special_requests):
    if not n_clicks:
        return ""

    # Crear diccionario con el mismo orden que las features del modelo
    new_reservation = {
        'no_of_adults': adults,
        'no_of_children': children,
        'no_of_weekend_nights': weekend_nights,
        'no_of_week_nights': week_nights,
        'type_of_meal_plan': meal_plan,
        'required_car_parking_space': car_parking,
        'room_type_reserved': room_type,
        'lead_time': lead_time,
        'arrival_year': year,
        'arrival_month': month,
        'arrival_date': date,
        'market_segment_type': market_segment,
        'repeated_guest': repeated_guest,
        'no_of_previous_cancellations': prev_cancel,
        'no_of_previous_bookings_not_canceled': prev_not_cancel,
        'no_of_special_requests': special_requests,
    }

    print("ejecuta aqui")
    
    # Crear DataFrame
    df_new = pd.DataFrame([new_reservation])

    # Asegurarse que columnas estén en el orden esperado
    df_new = df_new[state['feature_order']]

    # Predecir
    pred = state['model'].predict(df_new)[0]

    return f"Predicción de precio promedio por habitación: ${pred:.2f}"