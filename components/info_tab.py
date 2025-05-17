# from dash import  html
# from utils.read_file import read_file

# def info_tab(filepath):
#   if not filepath:
#       return html.Div("No hay datos cargados aún.")


#   try:
#       df = read_file(filepath)
#       return html.Div([
#           html.H5(f"Tamaño del DataFrame: {df.shape[0]} filas x {df.shape[1]} columnas")
#       ])
#   except Exception as e:
#       return html.Div(f"Error leyendo archivo: {str(e)}")


# components/info_tab.py
from dash import html
from utils.file_to_df import file_to_df

def info_tab(filepath):
    if not filepath:
        return html.Div("No se ha subido ningún archivo.")

    try:
        df = file_to_df(filepath)
    except Exception as e:
        return html.Div(f"Error al cargar archivo: {str(e)}")

    # Resumen de columnas
    columnas = html.Ul([html.Li(f"{col} ({dtype})") for col, dtype in zip(df.columns, df.dtypes)])

    # Conteo de nulos
    nulos = df.isnull().sum()
    nulos_list = html.Ul([html.Li(f"{col}: {nulos[col]} valores nulos") for col in df.columns if nulos[col] > 0])

    return html.Div([
        html.H4("Información general del archivo"),
        html.P(f"Filas: {df.shape[0]}"),
        html.P(f"Columnas: {df.shape[1]}"),
        html.H5("Columnas y tipos de datos:"),
        columnas,
        html.H5("Valores nulos:"),
        nulos_list if not nulos.empty else html.P("No hay valores nulos.")
    ])