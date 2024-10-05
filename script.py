import pandas as pd
import requests

# Cargar el archivo credit_pred.csv
file_path = './data/credit_pred.csv'
df = pd.read_csv(file_path)

# Seleccionar solo las columnas que el modelo espera
expected_columns = ['X1', 'X5', 'X12', 'X18', 'X19', 'X20', 'X21', 'X22', 'X23', 'X6', 'X2', 'X3', 'X4']
df_selected = df[expected_columns]  # Asegurarse de que solo se utilicen estas columnas

# URL de la API
api_url = 'http://127.0.0.1:1234/predict'

# Crear una lista para almacenar las predicciones
predictions = []

# Iterar sobre cada fila del DataFrame
for index, row in df_selected.iterrows():
    # Extraer las características de la fila como lista
    features = row.values.tolist()  # Convierte cada fila en una lista
    
    # Crear el payload para la solicitud
    payload = {
        "X": features
    }

    # Enviar la solicitud POST a la API
    response = requests.post(api_url, json=payload)
    
    # Si la solicitud fue exitosa, obtener la predicción
    if response.status_code == 200 and response.json() != -1:
        prediction = response.json().get("prediction")
        predictions.append(prediction)
    else:
        print(f"Error en la fila {index}: {response.status_code}, {response.text}")
        predictions.append(None)  # En caso de error, agregamos un valor None

# Agregar la columna 'Y' con las predicciones al DataFrame original
df['Y'] = predictions

# Guardar el DataFrame actualizado de nuevo en el archivo CSV
output_file_path = './data/xgb_credit_pred_with_predictions.csv'
df.to_csv(output_file_path, index=False)

print(f"Archivo guardado con las predicciones en: {output_file_path}")
