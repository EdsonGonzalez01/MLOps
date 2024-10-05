from fastapi import FastAPI, HTTPException
from sklearn.preprocessing import StandardScaler
import pickle
import pandas as pd
from pydantic import BaseModel

class InputData(BaseModel):
    X: list[float]

app = FastAPI()

# Cargar el modelo y el escalador
try:
    model = pickle.load(open("./models/XGBoost_model.pkl", "rb"))
    scaler = pickle.load(open("./models/xgb_scaler.pkl", "rb"))  # Cargar el escalador guardado
except Exception as e:
    raise RuntimeError(f"Error al cargar el modelo o escalador: {e}")

@app.get("/health")
def health_check():
    return {
        "status": "OK"
    }

@app.post("/predict")
async def predict(input_data: InputData):
    try:
        # Los nombres de características esperados (deben coincidir con la fase de entrenamiento)
        expected_columns = ['X1', 'X5', 'X12', 'X18', 'X19', 'X20', 'X21', 'X22', 'X23', 'X6', 'X2', 'X3', 'X4']
        
        # Verificar si la entrada tiene el número correcto de características
        if len(input_data.X) != len(expected_columns):
            raise HTTPException(status_code=400, detail="Número de características inválido")

        # Crear un DataFrame con los nombres de características esperados
        X_input = pd.DataFrame([input_data.X], columns=expected_columns)
        
        # Usar el escalador cargado para escalar la entrada
        X_input_scaled = scaler.transform(X_input)
        
        # Hacer una predicción
        prediction = model.predict(X_input_scaled)

        return {
            "prediction": int(prediction[0])
        }
    
    except Exception as e:
        # Manejar cualquier error en el proceso de predicción
        raise HTTPException(status_code=500, detail=f"Error en la predicción: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", port=1234, reload=True)
