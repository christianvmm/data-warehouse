# import io
# import base64
# import pandas as pd


# def file_to_df(contents, filename):
#     content_type, content_string = contents.split(',')
#     decoded = base64.b64decode(content_string)

#     if 'csv' in filename:
#       df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))

#     elif 'xls' in filename:
#       df = pd.read_excel(io.BytesIO(decoded))

#     elif 'json' in filename:
#       df = pd.read_json(io.StringIO(decoded.decode('utf-8')))

#     else:
#       raise ValueError("Formato no soportado")

#     return df


import pandas as pd

def file_to_df(filepath):
    if filepath.endswith('.csv'):
        return pd.read_csv(filepath)

    elif filepath.endswith(('.xls', '.xlsx')):
        return pd.read_excel(filepath)

    elif filepath.endswith('.json'):
        return pd.read_json(filepath)

    else:
        raise ValueError("Formato no soportado")