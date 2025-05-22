
#SEGUN YO NO SE OCUPA, SEGUN LO QUE DICE EN CLASSROOM, ASI QUE HAY Q QUITARLO PQ NO JALA

from dash import callback, Output, Input, State, html, dcc
import dash
import pandas as pd
import os
import plotly.express as px
from sklearn.linear_model import LinearRegression
import tempfile

TMP_DIR = os.path.join(tempfile.gettempdir(), 'dash_uploads')

@callback(
    Output('regression-variable-selectors', 'children'),
    Input('mining-technique-dropdown', 'value'),
    State('transformed-filepath', 'data'),
)
def mostrar_dropdowns_regresion(tecnica, processed_filename):
    if tecnica != 'regression' or not processed_filename:
        return []

    fullpath = os.path.join(TMP_DIR, processed_filename)
    if not os.path.exists(fullpath):
        return html.Div("Archivo no encontrado.")

    df = pd.read_csv(fullpath)
    df = df.loc[:, ~df.columns.duplicated()]
    
    numeric_cols = df.select_dtypes(include='number').columns.tolist()

    if len(numeric_cols) < 2:
        return html.Div("Se requieren al menos dos columnas numéricas para regresión.")

    return html.Div([
        html.Label("Selecciona columna objetivo (Y):"),
        dcc.Dropdown(id='regression-target-dropdown',
                     options=[{'label': col, 'value': col} for col in numeric_cols],
                     value=numeric_cols[0]),

        html.Label("Selecciona columnas predictoras (X):"),
        dcc.Dropdown(id='regression-features-dropdown',
                     options=[{'label': col, 'value': col} for col in numeric_cols if col != numeric_cols[0]],
                     multi=True,
                     value=numeric_cols[1:2] if len(numeric_cols) > 1 else [])
    ])


@callback(
    Output('mining-output-container', 'children'),
    Input('regression-target-dropdown', 'value'),
    Input('regression-features-dropdown', 'value'),
    State('mining-technique-dropdown', 'value'),
    State('transformed-filepath', 'data'),
)
def aplicar_tecnica_regresion(target_col, feature_cols, tecnica, processed_filename):
    if tecnica != 'regression':
        raise dash.exceptions.PreventUpdate

    if not target_col or not feature_cols or not processed_filename:
        return html.Div("Faltan datos para aplicar la técnica de regresión.")

    fullpath = os.path.join(TMP_DIR, processed_filename)
    if not os.path.exists(fullpath):
        return html.Div("Archivo no encontrado.")

    try:
        df = pd.read_csv(fullpath)
        df = df.loc[:, ~df.columns.duplicated()]
        X = df[feature_cols]
        y = df[target_col]

        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)

        fig = px.scatter(x=y, y=y_pred,
                         labels={'x': 'Valor Real', 'y': 'Valor Predicho'},
                         title='Regresión Lineal')
        fig.add_shape(type='line', x0=y.min(), y0=y.min(), x1=y.max(), y1=y.max(),
                      line=dict(color='red', dash='dash'))

        return html.Div([
            html.H5("Modelo de Regresión Lineal"),
            html.P(f"Coeficientes: {model.coef_}"),
            html.P(f"Intercepto: {model.intercept_}"),
            dcc.Graph(figure=fig)
        ])
    except Exception as e:
        return html.Div(f"Ocurrió un error durante la regresión: {str(e)}")