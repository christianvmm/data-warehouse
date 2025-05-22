import os
import base64
from utils.file_to_df import file_to_df
from utils.file_to_df import file_to_df

def read_file_for_resultados(filepath):
    # Simplemente delega a file_to_df pasando el path
    return file_to_df(filepath)
