import uuid
import os
import base64

TMP_DIR = '/tmp/dash_uploads'

def save_file(contents, filename):
    # contents viene en base64: "data:;base64,ABC123..."
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    
    # Para evitar colisiones, nombre único
    unique_filename = f"{uuid.uuid4()}_{filename}"
    filepath = os.path.join(TMP_DIR, unique_filename)
    
    with open(filepath, "wb") as f:
        f.write(decoded)
    return filepath