import uuid
import os
import base64
import tempfile


TMP_DIR = os.path.join(tempfile.gettempdir(), 'dash_uploads')
os.makedirs(TMP_DIR, exist_ok=True)

def save_file(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)

    ext = os.path.splitext(filename)[1]
    safe_filename = filename.replace(" ", "_")  # evitar espacios
    unique_filename = f"{str(uuid.uuid4())}_{safe_filename}"
    filepath = os.path.join(TMP_DIR, unique_filename)

    with open(filepath, 'wb') as f:
        f.write(decoded)

    return filepath