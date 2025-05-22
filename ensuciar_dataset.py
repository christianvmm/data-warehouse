import pandas as pd
import numpy as np
import random

# Leer el archivo original
df = pd.read_csv("Hotel_Reservations.csv")

# 1. Repetir algunas filas (10% duplicados)
duplicates = df.sample(frac=0.1, random_state=1)
df_dirty = pd.concat([df, duplicates], ignore_index=True)

# 2. Introducir valores nulos aleatorios en columnas seleccionadas
cols_to_nullify = ['no_of_children', 'type_of_meal_plan', 'room_type_reserved', 'lead_time', 'avg_price_per_room']
for col in cols_to_nullify:
    indices = df_dirty.sample(frac=0.1, random_state=random.randint(0, 1000)).index
    df_dirty.loc[indices, col] = np.nan

# 3. Crear una columna con la fecha combinada en formato sucio
def format_date(row):
    try:
        if random.random() < 0.3:
            return f"{int(row['arrival_year']):04d}{int(row['arrival_month']):02d}{int(row['arrival_date']):02d}"
        else:
            return f"{int(row['arrival_year'])}-{int(row['arrival_month']):02d}-{int(row['arrival_date']):02d}"
    except:
        return np.nan  # Por si hay NaNs en la fecha original

df_dirty['arrival_full_date'] = df_dirty.apply(format_date, axis=1)

# 4. Asegurarse que 'repeated_guest' solo tenga 0 o 1
df_dirty['repeated_guest'] = df_dirty['repeated_guest'].apply(lambda x: 0 if x == 0 else 1)

# 5. Guardar el archivo ensuciado
df_dirty.to_csv("hoteles_sucio.csv", index=False)

print("✅ Archivo 'hoteles_sucio.csv' generado con éxito desde Hotel_Reservations.csv")