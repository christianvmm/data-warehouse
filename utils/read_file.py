import os
import base64
from utils.file_to_df import file_to_df

def read_file(filepath):
    # Aquí lees el archivo guardado y conviertes a DataFrame
    # Asumiendo que file_to_df pueda leer desde filepath o adaptas:
    with open(filepath, "rb") as f:
        content = f.read()
    # file_to_df espera contenido en base64 + nombre + fecha?
    # Entonces recreamos contenido base64 para usar file_to_df
    b64_content = "data:;base64," + base64.b64encode(content).decode()
    
    filename = os.path.basename(filepath).split("_",1)[1]  # quitar uuid
    date = os.path.getmtime(filepath)
    df = file_to_df(b64_content, filename, date)
    return df