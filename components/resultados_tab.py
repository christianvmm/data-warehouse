from dash import html, dcc, Input, Output, State, callback
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix
import plotly.figure_factory as ff

from utils.read_file_for_resultados import read_file_for_resultados

# Variable global
global_df = pd.DataFrame()

def resultados_tab(filepath):
    global global_df
    if not filepath:
        return dbc.Container([
            dbc.Alert("No hay datos cargados aún.", color="secondary", className="mt-4")
        ], fluid=True)

    try:
        df = read_file_for_resultados(filepath)
        if df is None:
            return px.imshow([[0]], labels=dict(x="Error al leer archivo", y="", color=""))

        if 'repeated_guest' not in df.columns:
            return dbc.Container([
                dbc.Alert("La columna 'repeated_guest' no está presente en los datos.", color="danger", className="mt-4")
            ], fluid=True)

        global_df = df

        columnas_validas = df.select_dtypes(include=['int64', 'float64', 'object']).columns.tolist()
        columnas_validas = [col for col in columnas_validas if col != 'repeated_guest']

        if len(columnas_validas) < 2:
            return dbc.Container([
                dbc.Alert("No hay suficientes columnas válidas para análisis.", color="warning", className="mt-4")
            ], fluid=True)

        return dbc.Container([
            dbc.Card(
                dbc.CardBody([
                    html.H4("Análisis de probabilidad de regreso", className="text-primary mb-3"),
                    html.P(f"Tamaño del dataframe: {df.shape[0]} filas × {df.shape[1]} columnas"),

                    dbc.Row([
                        dbc.Col([
                            html.Label("Eje X (columna):"),
                            dcc.Dropdown(
                                id="x-variable",
                                options=[{"label": col, "value": col} for col in columnas_validas],
                                value=columnas_validas[0],
                                clearable=False
                            )
                        ], md=6),
                        dbc.Col([
                            html.Label("Eje Y (columna):"),
                            dcc.Dropdown(
                                id="y-variable",
                                options=[{"label": col, "value": col} for col in columnas_validas],
                                value=columnas_validas[1] if len(columnas_validas) > 1 else columnas_validas[0],
                                clearable=False
                            )
                        ], md=6)
                    ], className="mt-3"),

                    html.Div(id="heatmap-container", className="mt-4"),

                    html.Hr(className="my-4"),

                    html.H5("Modelo de predicción de regreso"),
                    dbc.Row([
                        dbc.Col([
                            html.Label("Seleccionar modelo:"),
                            dcc.Dropdown(
                                id="modelo-selector",
                                options=[
                                    {"label": "Regresión logística", "value": "logistic"},
                                    {"label": "Árbol de decisión", "value": "tree"}
                                ],
                                value="logistic",
                                clearable=False
                            )
                        ], md=6),
                        dbc.Col([
                            html.Br(),
                            dbc.Button("Entrenar modelo", id="btn-entrenar", color="success", className="mt-2", n_clicks=0)
                        ], md=6)
                    ]),

                    html.Div(id="modelo-resultados", className="mt-4")
                ]),
                className="shadow-sm rounded mt-4 p-3"
            )
        ], fluid=True, style={'maxWidth': '1000px'})

    except Exception as e:
        return dbc.Container([
            dbc.Alert(f"Error leyendo archivo: {str(e)}", color="danger", className="mt-4")
        ], fluid=True)

# Callback para el heatmap
@callback(
    Output("heatmap-container", "children"),
    Input("x-variable", "value"),
    Input("y-variable", "value")
)
def update_heatmap(x_col, y_col):
    if global_df.empty or not x_col or not y_col:
        return dbc.Alert("Datos insuficientes para generar la gráfica.", color="warning")

    try:
        pivot_table = global_df.groupby([y_col, x_col])['repeated_guest'].mean().unstack().fillna(0)
        fig = go.Figure(data=go.Heatmap(
            z=pivot_table.values,
            x=pivot_table.columns,
            y=pivot_table.index,
            colorscale='YlGnBu',
            hovertemplate=f'{x_col}: %{{x}}<br>{y_col}: %{{y}}<br>Probabilidad: %{{z:.2f}}<extra></extra>'
        ))
        fig.update_layout(
            title=f'Probabilidad de regreso: {y_col} vs {x_col}',
            xaxis_title=x_col,
            yaxis_title=y_col,
            margin=dict(l=50, r=30, t=50, b=50),
            height=600
        )
        return dcc.Graph(figure=fig)

    except Exception as e:
        return dbc.Alert(f"Error generando gráfica: {str(e)}", color="danger")


# Callback para entrenamiento del modelo
@callback(
    Output("modelo-resultados", "children"),
    Input("btn-entrenar", "n_clicks"),
    State("modelo-selector", "value")
)
def entrenar_modelo(n_clicks, modelo_seleccionado):
    if n_clicks == 0 or global_df.empty:
        return ""

    try:
        df_model = global_df.copy().dropna()

        y = df_model["repeated_guest"]
        X = df_model.drop(columns=["repeated_guest"])

        # Limitar columnas categóricas con demasiadas categorías
        cat_cols = X.select_dtypes(include='object').columns
        low_cardinality_cols = [col for col in cat_cols if X[col].nunique() <= 50]

        # Excluir columnas categóricas con alta cardinalidad
        high_card_cols = [col for col in cat_cols if col not in low_cardinality_cols]
        if high_card_cols:
            X = X.drop(columns=high_card_cols)

        # Aplicar get_dummies solo a columnas con cardinalidad baja
        X = pd.get_dummies(X, columns=low_cardinality_cols, drop_first=True)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        if modelo_seleccionado == "logistic":
            model = LogisticRegression(max_iter=1000)
        else:
            model = DecisionTreeClassifier(max_depth=5)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        fig_cm = ff.create_annotated_heatmap(
            z=cm,
            x=["No Regresa", "Regresa"],
            y=["No Regresa", "Regresa"],
            colorscale="Viridis"
        )
        fig_cm.update_layout(title="Matriz de Confusión", margin=dict(t=40))

        return [
            html.P(f"Exactitud del modelo: {acc:.2%}", className="text-success fw-bold"),
            dcc.Graph(figure=fig_cm),
            html.Div([
                html.H6("Importancia de variables (si aplica):"),
                dcc.Graph(
                    figure=px.bar(
                        x=model.feature_importances_ if modelo_seleccionado == "tree" else model.coef_[0],
                        y=X.columns,
                        orientation='h',
                        labels={'x': 'Importancia', 'y': 'Variable'},
                        title="Importancia de características"
                    )
                )
            ])
        ]
    except Exception as e:
        return dbc.Alert(f"Error al entrenar modelo: {str(e)}", color="danger")

