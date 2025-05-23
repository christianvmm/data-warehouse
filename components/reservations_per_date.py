import os
import tempfile
import pandas as pd
import plotly.graph_objects as go
from dash import html, dcc, dash_table
import dash_bootstrap_components as dbc

def reservations_per_date(fullpath):
    # Carga datos
    df = pd.read_csv(fullpath)

    # Convertir a datetime con manejo de errores
    df['arrival_date'] = pd.to_datetime({
        'year': df['arrival_year'],
        'month': df['arrival_month'],
        'day': df['arrival_date']
    }, errors='coerce')

    # Eliminar filas con fechas inválidas
    df = df.dropna(subset=['arrival_date'])

    # Agrupar por mes
    reservas_por_mes = df.groupby(df['arrival_date'].dt.to_period('M')).size()
    precio_promedio_mes = df.groupby(df['arrival_date'].dt.to_period('M'))['avg_price_per_room'].mean()

    # Convertir índice Period a string
    reservas_por_mes.index = reservas_por_mes.index.astype(str)
    precio_promedio_mes.index = precio_promedio_mes.index.astype(str)

    # Crear figura con doble eje Y
    fig = go.Figure()

    # Barras: número de reservas
    fig.add_trace(go.Bar(
        x=reservas_por_mes.index,
        y=reservas_por_mes.values,
        name='Número de Reservas',
        marker_color='lightskyblue',
        yaxis='y1'
    ))

    # Línea: precio promedio
    fig.add_trace(go.Scatter(
        x=precio_promedio_mes.index,
        y=precio_promedio_mes.values,
        name='Precio Promedio',
        mode='lines+markers',
        line=dict(color='tomato', width=3),
        yaxis='y2'
    ))

    # Layout de la figura
    fig.update_layout(
        title='Número de Reservas y Precio Promedio por Mes',
        xaxis=dict(title='Mes'),
        yaxis=dict(title='Número de Reservas', side='left'),
        yaxis2=dict(title='Precio Promedio', overlaying='y', side='right'),
        legend=dict(x=0.01, y=0.99),
        margin=dict(l=40, r=40, t=60, b=40),
        template='plotly_white',
        height=500
    )

    # Retornar componente de Dash
    return dbc.Card([
        dbc.CardHeader(html.H5("Visualizar Reservas por Época del Año")),
        dbc.CardBody([
            dcc.Graph(figure=fig)
        ])
    ], className="mb-4 shadow-sm")
