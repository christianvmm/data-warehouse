# from dash import html
# from utils.read_file import read_file
# from dash import dash_table


# def upload_tab(filepath):
#     children = []

#     if filepath:
#         try:
#             df = read_file(filepath)
#             children.append(render_table(df))
#         except Exception as e:
#             children.append(html.Div(f"Error leyendo archivo: {str(e)}"))

#     return children


# def render_table(df):
#     return dash_table.DataTable(
#         df.to_dict('records'),
#         [{'name': i, 'id': i} for i in df.columns],
#         style_table={'overflowX': 'auto'},
#         page_size=10
#     )


from dash import html, dash_table
from utils.file_to_df import file_to_df

def upload_tab(filepath):
    if not filepath:
        return html.Div("No se ha subido ningún archivo aún.")
    
    try:
        df = file_to_df(filepath)
    except Exception as e:
        return html.Div(f"Error al leer el archivo: {str(e)}")
    
    return html.Div([
        html.H4("Vista preliminar del archivo:"),
        dash_table.DataTable(
            data=df.head(10).to_dict('records'),
            columns=[{'name': col, 'id': col} for col in df.columns],
            page_size=10,
            style_table={'overflowX': 'auto'}
        )
    ])