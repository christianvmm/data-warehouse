from dash import callback, Output, Input, State, html, dcc
import dash
import pandas as pd
import os
import plotly.express as px
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import tempfile

TMP_DIR = os.path.join(tempfile.gettempdir(), 'dash_uploads')


@callback(
    Output('classification-variable-selectors', 'children'),
    Input('mining-technique-dropdown', 'value'),
    State('transformed-filepath', 'data'),
)
def mostrar_dropdowns_clasificacion(tecnica, processed_filename):
    if tecnica != 'classification' or not processed_filename:
        return []

    fullpath = os.path.join(TMP_DIR, processed_filename)
    if not os.path.exists(fullpath):
        return html.Div("Archivo no encontrado.")

    df = pd.read_csv(fullpath)
    df = df.loc[:, ~df.columns.duplicated()]
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    categorical_cols = df.select_dtypes(include='object').columns.tolist()

    if not categorical_cols or not numeric_cols:
        return html.Div("No se encontraron columnas adecuadas para clasificación.")

    return html.Div([
        html.Label("Selecciona columna objetivo (Y - categórica):"),
        dcc.Dropdown(id='classification-target-dropdown',
                     options=[{'label': col, 'value': col} for col in categorical_cols],
                     value=categorical_cols[0]),

        html.Label("Selecciona columnas predictoras (X - numéricas):"),
        dcc.Dropdown(id='classification-features-dropdown',
                     options=[{'label': col, 'value': col} for col in numeric_cols],
                     multi=True,
                     value=numeric_cols[:2] if len(numeric_cols) >= 2 else numeric_cols)
    ])


@callback(
    Output('mining-output-container', 'children', allow_duplicate=True),
    Input('classification-target-dropdown', 'value'),
    Input('classification-features-dropdown', 'value'),
    State('mining-technique-dropdown', 'value'),
    State('transformed-filepath', 'data'),
    prevent_initial_call = True
)
def aplicar_tecnica_clasificacion(target_col, feature_cols, tecnica, processed_filename):
    if tecnica != 'classification':
        raise dash.exceptions.PreventUpdate

    if not target_col or not feature_cols or not processed_filename:
        return html.Div("Faltan datos para aplicar la técnica de clasificación.")

    fullpath = os.path.join(TMP_DIR, processed_filename)
    if not os.path.exists(fullpath):
        return html.Div("Archivo no encontrado.")

    df = pd.read_csv(fullpath)
    df = df.loc[:, ~df.columns.duplicated()]

    try:
        X = df[feature_cols]
        y = df[target_col]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        model = DecisionTreeClassifier()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        return html.Div([
            html.H5("Árbol de Decisión - Clasificación"),
            html.P(f"Precisión del modelo: {acc:.2f}"),
            html.Pre(f"Matriz de confusión:\n{cm}")
        ])
    except Exception as e:
        return html.Div(f"Ocurrió un error durante la clasificación: {str(e)}")