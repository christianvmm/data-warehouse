import os
import io
import uuid
from dash import Dash, dcc, html, dash_table, Input, Output, State, callback, no_update
import dash
from dash.exceptions import PreventUpdate
import datetime
import pandas as pd
import base64
import plotly.express as px
from utils.file_to_df import file_to_df
from utils.read_file import read_file
from utils.save_file import save_file
from components.upload_tab import upload_tab
from components.info_tab import info_tab
from components.etl_tab import etl_tab
from components.mineria_tab import mineria_tab
from components.resultados_tab import resultados_tab
import tempfile

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt


TMP_DIR = os.path.join(tempfile.gettempdir(), 'dash_uploads')
os.makedirs(TMP_DIR, exist_ok=True) # Asegura que exista la carpeta

# TMP_DIR = '/tmp/dash_uploads'
# os.makedirs(TMP_DIR, exist_ok=True)  # Asegura que exista la carpeta


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# límite de subida (para archivos > 16MB)
app = Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)
app.server.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB

app.layout = html.Div([
    dcc.Store(id='transformed-filepath'), 
    dcc.Store(id='stored-filename'),  # Guardar solo el nombre del archivo
    
    
    html.Div([
        html.H3("Sube un archivo (.csv, .xlsx, .json):"),
        dcc.Upload(
            id='upload-data',
            children=html.Button('Subir Archivo'),
            multiple=False
        )
    ], style={'textAlign': 'center', 'marginBottom': '20px'}),

    dcc.Tabs(id='tabs', value='tab-upload', children=[
        dcc.Tab(label='Subir y ver datos', value='tab-upload'),
        dcc.Tab(label='Información', value='tab-info'),
        dcc.Tab(label='ETL', value='tab-etl'),
        dcc.Tab(label='Estadisticas Descriptivas y Minería de datos', value='tab-mineria'),
        dcc.Tab(label='Resultados', value='tab-resultados'),
    ]),
    
    html.Div(id='tab-content')
    
])


# Callback para guardar el archivo subido
@callback(
    Output('stored-filename', 'data'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),
    prevent_initial_call=True
)
def save_uploaded_file(contents, filename):
    if contents is None:
        return no_update
    filepath = save_file(contents, filename)
    # print("Archivo guardado en:", filepath)
    return filepath




# Callback para ETL
@callback(
    Output('etl-table', 'children'),
    Output('etl-feedback', 'children'),
    Output('transformed-filepath', 'data'),
    Input('etl-apply-button', 'n_clicks'),
    # Limpieza
    State('etl-drop-columns', 'value'),
    State('etl-convert-date', 'value'),
    State('etl-options', 'value'), 
    # Transformacion
    State('etl-normalize-columns', 'value'),
    State('etl-filter-column', 'value'),
    State('etl-filter-min', 'value'),
    State('etl-filter-max', 'value'),
    State('stored-filename', 'data'),
    
    prevent_initial_call=True
)
def aplicar_etl(n_clicks, cols_to_drop, cols_to_date, etl_options,
              normalize_cols, filter_col, filter_min, filter_max, filepath):
    if filepath is None:
        return no_update, "No hay archivo cargado."

    df = file_to_df(filepath)
    feedback_msgs = []

    # 1. Eliminar columnas
    if cols_to_drop:
        df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
        feedback_msgs.append(f"Eliminadas {len(cols_to_drop)} columnas.")

    # 2. Convertir fechas
    if cols_to_date:
        for col in cols_to_date:
            try:
                # Formatos de YYYY/MM/DD o YYYYMMDD, cambian a YYYY-MM-DD
                # YYYY/MM/DD
                df[col] = df[col].astype(str).str.replace('/', '-').str.strip()

                # YYYYMMDD
                def fix_yyyymmdd(fecha):
                    fecha = fecha.strip()
                    if len(fecha) == 8 and fecha.isdigit():
                        return fecha[:4] + '-' + fecha[4:6] + '-' + fecha[6:]
                    return fecha

                df[col] = df[col].apply(fix_yyyymmdd)

                df[col] = pd.to_datetime(df[col], errors='coerce')
                df[col] = df[col].dt.strftime('%Y-%m-%d')
                feedback_msgs.append(f"'{col}' convertido a formato YYYY-MM-DD.")
            except Exception as e:
                feedback_msgs.append(f"Error al convertir '{col}' a fecha: {str(e)}")

    # 3. Reemplazar nulos 
    if 'nulls' in etl_options:
        for col in df.columns:
            if df[col].isnull().any():
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col].fillna(0, inplace=True)
                    feedback_msgs.append(f"Nulos en '{col}' reemplazados por 0.")
                else:
                    df[col].fillna('Desconocido', inplace=True)
                    feedback_msgs.append(f"Nulos en '{col}' reemplazados por 'Desconocido'.")

    # 4. Eliminar duplicados
    if 'duplicates' in etl_options:
        before = len(df)
        df.drop_duplicates(inplace=True)
        after = len(df)
        removed = before - after
        feedback_msgs.append(f"Eliminadas {removed} filas duplicadas.")


     # 5. Normalización
    if normalize_cols:
        for col in normalize_cols:
            if pd.api.types.is_numeric_dtype(df[col]):
                min_val = df[col].min()
                max_val = df[col].max()
                if max_val != min_val:
                    df[col] = (df[col] - min_val) / (max_val - min_val)
                    feedback_msgs.append(f"Columna '{col}' normalizada.")
                else:
                    feedback_msgs.append(f"No se normalizó '{col}' (valor constante).")

    # 6 . Filtrado 
    if filter_col and (filter_min is not None or filter_max is not None):
        original_len = len(df)
        if filter_min is not None:
            df = df[df[filter_col] >= filter_min]
        if filter_max is not None:
            df = df[df[filter_col] <= filter_max]
        feedback_msgs.append(f"Filtrado aplicado en '{filter_col}'. Filas reducidas de {original_len} a {len(df)}.")

    # Guardar el dataframe transformado en un archivo temporal
    filename = f"processed_{uuid.uuid4().hex}.csv"
    fullpath = os.path.join(TMP_DIR, filename)
    df.to_csv(fullpath, index=False)

    # Mostrar tabla
    table = dash_table.DataTable(
        data=df.head(20).to_dict('records'),
        columns=[{'name': i, 'id': i} for i in df.columns],
        style_table={'overflowX': 'auto'},
        style_cell={'textAlign': 'left'}
    )

    # feedback = f"Archivo procesado guardado: {filename}"

    # return table, html.Ul([html.Li(msg) for msg in feedback_msgs]), df.to_json(date_format='iso', orient='split')
    # return (
    #     table,
    #     html.Ul([html.Li(msg) for msg in feedback_msgs]),
    #     {
    #         'columns': df.columns.tolist(),
    #         'data': df.to_dict('records')
    #     }
    # )
    return table, html.Ul([html.Li(msg) for msg in feedback_msgs]), filename

# Callback para descargar el archivo transformado
@callback(
    Output("download-data", "data"),
    Input("download-button", "n_clicks"),
    State("transformed-df", "data"),
    State("download-format", "value"),
    prevent_initial_call=True
)
def download_file(n_clicks, df_json, format_selected):
    if df_json is None:
        raise PreventUpdate

    df = pd.read_json(df_json, orient='split')

    if format_selected == "csv":
        return dcc.send_bytes(df.to_csv(index=False).encode(), "datos_transformados.csv")

    elif format_selected == "json":
        return dcc.send_bytes(df.to_json(orient="records", indent=2).encode(), "datos_transformados.json")

    elif format_selected == "excel":
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Datos')
        buffer.seek(0)
        return dcc.send_bytes(buffer.read(), "datos_transformados.xlsx")

    else:
        raise PreventUpdate


# Callback para graficos
@callback(
    Output('eda-plots-container', 'children'),
    Input('eda-numeric-dropdown', 'value'),
    State('transformed-filepath', 'data'),
    prevent_initial_call=True
)
def update_eda_graficos(selected_col, processed_filename):
    if selected_col is None or processed_filename is None:
        return html.Div("No hay columna seleccionada o datos procesados.")

    fullpath = os.path.join(TMP_DIR, processed_filename)
    if not os.path.exists(fullpath):
        return html.Div("Archivo procesado no encontrado.")

    df = pd.read_csv(fullpath)

    fig_hist = px.histogram(df, x=selected_col, nbins=30, title=f"Histograma de Columna '{selected_col}'")
    fig_box = px.box(df, y=selected_col, title=f"Boxplot de Columna '{selected_col}'")

    return html.Div([
        dcc.Graph(figure=fig_hist),
        dcc.Graph(figure=fig_box)
    ])


@callback(
    Output('cluster-variable-selectors', 'children'),
    Input('mining-technique-dropdown', 'value'),
    State('transformed-filepath', 'data'),
)
def actualizar_dropdowns(tecnica, processed_filename):
    if tecnica == 'kmeans':
        return mostrar_dropdowns_cluster(tecnica, processed_filename)
    elif tecnica == 'decision_tree':
        return mostrar_dropdowns_classification(tecnica, processed_filename)
    elif tecnica == 'regression':
        return mostrar_dropdowns_regresion(tecnica, processed_filename)
    return html.Div()


# Callback para tecnicas de datamining
# @callback(
#     Output('mining-output-container', 'children'),
#     [
#         Input('mining-technique-dropdown', 'value'),
#         Input('cluster-x-dropdown', 'value'),
#         Input('cluster-y-dropdown', 'value'),
#         Input('target-column-dropdown', 'value'),
#         Input('feature-columns-dropdown', 'value'),
#         Input('regression-x-dropdown', 'value'),
#         Input('regression-y-dropdown', 'value'),
#     ],
#     State('transformed-filepath', 'data'),
#     prevent_initial_call=True
# )
# def aplicar_tecnica(tecnica, x_cluster, y_cluster, target_col, feature_cols, x_reg, y_reg, processed_filename):
#     ctx = dash.callback_context

#     if not processed_filename:
#         return html.Div("No hay archivo disponible.")

#     df = pd.read_csv(os.path.join(TMP_DIR, processed_filename))
#     df = df.loc[:, ~df.columns.duplicated()]

#     if tecnica == 'kmeans':
#         if not x_cluster or not y_cluster:
#             return html.Div("Selecciona columnas X e Y para clustering.")
#         X = df[[x_cluster, y_cluster]].dropna()
#         model = KMeans(n_clusters=3).fit(X)
#         X['cluster'] = model.labels_
#         fig = px.scatter(X, x=x_cluster, y=y_cluster, color=X['cluster'].astype(str))
#         return dcc.Graph(figure=fig)

#     elif tecnica == 'decision_tree':
#         if not target_col or not feature_cols:
#             return html.Div("Selecciona columnas para clasificación.")
#         df = df.dropna(subset=[target_col] + feature_cols)
#         le = LabelEncoder()
#         y = le.fit_transform(df[target_col])
#         X = df[feature_cols]
#         model = DecisionTreeClassifier(max_depth=3)
#         model.fit(X, y)
#         fig, ax = plt.subplots(figsize=(12, 6))
#         plot_tree(model, feature_names=feature_cols, class_names=le.classes_, filled=True, ax=ax)
#         buf = io.BytesIO()
#         plt.savefig(buf, format="png")
#         buf.seek(0)
#         img_encoded = base64.b64encode(buf.read()).decode('utf-8')
#         return html.Div([html.Img(src=f'data:image/png;base64,{img_encoded}')])

#     elif tecnica == 'regression':
#         if not x_reg or not y_reg:
#             return html.Div("Selecciona variables para regresión.")
#         df = df[[x_reg, y_reg]].dropna()
#         model = LinearRegression()
#         model.fit(df[[x_reg]], df[y_reg])
#         df['pred'] = model.predict(df[[x_reg]])
#         fig = px.scatter(df, x=x_reg, y=y_reg)
#         fig.add_traces(px.line(df, x=x_reg, y='pred').data)
#         return dcc.Graph(figure=fig)

#     return html.Div("Técnica no reconocida.")

@callback(
    Output('mining-output-container', 'children'),
    [
        Input('cluster-x-dropdown', 'value'),
        Input('cluster-y-dropdown', 'value'),
        Input('target-column-dropdown', 'value'),
        Input('feature-columns-dropdown', 'value'),
        Input('regression-x-dropdown', 'value'),
        Input('regression-y-dropdown', 'value'),
        State('mining-technique-dropdown', 'value'),
        State('transformed-filepath', 'data'),
    ],
    prevent_initial_call=True
)
def aplicar_tecnica_general(x_cluster, y_cluster, target_col, feature_cols, x_reg, y_reg, tecnica, processed_filename):
    if not processed_filename or not tecnica:
        return html.Div("No hay datos o técnica seleccionada.")

    if tecnica == 'kmeans':
        return aplicar_tecnica_cluster(x_cluster, y_cluster, tecnica, processed_filename)

    elif tecnica == 'decision_tree':
        return aplicar_tecnica_clasificacion(target_col, feature_cols, tecnica, processed_filename)

    elif tecnica == 'regression':
        return aplicar_tecnica_regresion(x_reg, y_reg, tecnica, processed_filename)

    return html.Div("Técnica no reconocida.")



#1 OPCION DE MINERA | APLICAR CLUSTER

# DROPDOWN PARA MOSTRAR EL CLUSTER
def mostrar_dropdowns_cluster(tecnica, processed_filename):
    if tecnica != 'kmeans' or not processed_filename:
        return []

    fullpath = os.path.join(TMP_DIR, processed_filename)
    if not os.path.exists(fullpath):
        return html.Div("Archivo no encontrado.")

    df = pd.read_csv(fullpath)
    df = df.loc[:, ~df.columns.duplicated()]  # Eliminamos columnas duplicadas

    numeric_cols = df.select_dtypes(include='number').columns.tolist()

    if len(numeric_cols) < 2:
        return html.Div("No hay suficientes columnas numéricas para clustering.")

    options = [{'label': col, 'value': col} for col in numeric_cols]

    return html.Div([
        html.Label("Selecciona columna X:"),
        dcc.Dropdown(id='cluster-x-dropdown', options=options, value=numeric_cols[0]),
        html.Label("Selecciona columna Y:"),
        dcc.Dropdown(id='cluster-y-dropdown', options=options, value=numeric_cols[1])
    ])


# APLICAR CLUSTER
def aplicar_tecnica_cluster(x_col, y_col, tecnica, processed_filename):
    if tecnica != 'kmeans':
        return html.Div("Selecciona una técnica válida.")

    if not x_col or not y_col or not processed_filename:
        return html.Div("Faltan columnas o archivo.")

    fullpath = os.path.join(TMP_DIR, processed_filename)
    if not os.path.exists(fullpath):
        return html.Div("Archivo no encontrado.")

    df = pd.read_csv(fullpath)
    df = df.loc[:, ~df.columns.duplicated()]  # Eliminar columnas duplicadas

    if x_col not in df.columns or y_col not in df.columns:
        return html.Div("Columnas inválidas.")

    X = df[[x_col, y_col]].dropna()

    from sklearn.cluster import KMeans
    import plotly.express as px

    kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
    X['cluster'] = kmeans.labels_

    fig = px.scatter(
        X, x=x_col, y=y_col, color=X['cluster'].astype(str),
        title='Clustering con K-Means',
        color_discrete_sequence=px.colors.qualitative.Set1
    )

    return dcc.Graph(figure=fig)



#2 OPCION DE MINERA | APLICAR CLASIFICACION

# DROPDOWN PARA CLASIFICACION
def mostrar_dropdowns_classification(tecnica, processed_filename):
    if tecnica != 'decision_tree' or not processed_filename:
        return []

    fullpath = os.path.join(TMP_DIR, processed_filename)
    if not os.path.exists(fullpath):
        return html.Div("Archivo no encontrado.")

    df = pd.read_csv(fullpath)
    df = df.loc[:, ~df.columns.duplicated()]

    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    categorical_cols = df.select_dtypes(include='object').columns.tolist()

    if not numeric_cols or not categorical_cols:
        return html.Div("No hay suficientes columnas para clasificación.")

    return html.Div([
        html.Label("Selecciona columna objetivo (categoría):"),
        dcc.Dropdown(id='target-column-dropdown', options=[{'label': col, 'value': col} for col in categorical_cols], value=categorical_cols[0]),
        html.Label("Selecciona variables predictoras:"),
        dcc.Dropdown(id='feature-columns-dropdown', options=[{'label': col, 'value': col} for col in numeric_cols], value=numeric_cols[:2], multi=True)
    ])

# APLICAR CLASIFICACION
def aplicar_tecnica_clasificacion(target_col, feature_cols, tecnica, processed_filename):
    if tecnica != 'decision_tree':
        return html.Div("Técnica no válida.")

    if not target_col or not feature_cols or not processed_filename:
        return html.Div("Faltan datos.")

    fullpath = os.path.join(TMP_DIR, processed_filename)
    if not os.path.exists(fullpath):
        return html.Div("Archivo no encontrado.")

    import pandas as pd
    from sklearn.tree import DecisionTreeClassifier, plot_tree
    from sklearn.preprocessing import LabelEncoder
    import matplotlib.pyplot as plt
    import plotly.tools as tls
    import io
    import base64

    df = pd.read_csv(fullpath)
    df = df.loc[:, ~df.columns.duplicated()]
    df = df.dropna(subset=[target_col] + feature_cols)

    le = LabelEncoder()
    y = le.fit_transform(df[target_col])
    X = df[feature_cols]

    clf = DecisionTreeClassifier(max_depth=3)
    clf.fit(X, y)

    # Graficar árbol
    fig, ax = plt.subplots(figsize=(12, 6))
    plot_tree(clf, feature_names=feature_cols, class_names=le.classes_, filled=True, ax=ax)
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    encoded_image = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    plt.close(fig)

    return html.Div([
        html.H5("Árbol de Decisión:"),
        html.Img(src='data:image/png;base64,{}'.format(encoded_image))
    ])



#3 OPCION DE MINERA | APLICAR REGRESION

# DROPDOWN PARA REGRESION
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
        return html.Div("Se requieren al menos 2 columnas numéricas.")

    return html.Div([
        html.Label("Variable independiente (X):"),
        dcc.Dropdown(id='regression-x-dropdown', options=[{'label': col, 'value': col} for col in numeric_cols], value=numeric_cols[0]),
        html.Label("Variable dependiente (Y):"),
        dcc.Dropdown(id='regression-y-dropdown', options=[{'label': col, 'value': col} for col in numeric_cols if col != numeric_cols[0]], value=numeric_cols[1])
    ])


# CALLBACK PARA LA REGRESION
def aplicar_tecnica_regresion(x_col, y_col, tecnica, processed_filename):
    if tecnica != 'regression':
        return html.Div("Técnica no válida.")

    if not x_col or not y_col or not processed_filename:
        return html.Div("Faltan datos.")

    import pandas as pd
    import plotly.express as px
    from sklearn.linear_model import LinearRegression
    import numpy as np

    fullpath = os.path.join(TMP_DIR, processed_filename)
    if not os.path.exists(fullpath):
        return html.Div("Archivo no encontrado.")

    df = pd.read_csv(fullpath)
    df = df.loc[:, ~df.columns.duplicated()]
    df = df[[x_col, y_col]].dropna()

    X = df[[x_col]].values
    y = df[y_col].values

    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    df['Predicción'] = y_pred

    fig = px.scatter(df, x=x_col, y=y_col, title="Regresión Lineal")
    fig.add_traces(px.line(df, x=x_col, y='Predicción').data)

    return dcc.Graph(figure=fig)


# Callback para renderizar el contenido de la pestaña seleccionada
@callback(
    Output('tab-content', 'children'),
    Input('tabs', 'value'),
    Input('stored-filename', 'data'),
    State('transformed-filepath', 'data')
)
def render_tab(tab, filepath, processed_filename):
    # print("Callback de render_tab activado")
    if tab == 'tab-upload':
        return upload_tab(filepath)
    
    elif tab == 'tab-info':
        return info_tab(filepath)
    
    elif tab == 'tab-etl':
        return etl_tab(filepath)
    
    elif tab == 'tab-mineria':
        # print("DEBUG: df_dict recibido en minería =", df_dict) 
        print("DEBUG - Archivo procesado que se pasa:", processed_filename)
        return mineria_tab(processed_filename)
    
    elif tab == 'tab-resultados':
        return resultados_tab(filepath)
    
    
        
if __name__ == '__main__':
    app.run(debug=True)
