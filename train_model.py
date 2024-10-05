import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
import pickle

# Cargar datos
data = pd.read_csv("./data/credit_train.csv")

# Selección de características basada en los resultados de las pruebas chi2 y ks
# Manteniendo las características que mostraron diferencias significativas
selected_features = ['X1', 'X5', 'X12', 'X18', 'X19', 'X20', 'X21', 'X22', 'X23', 'X6', 'X2', 'X3', 'X4']

X = data[selected_features]
Y = data["Y"]

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=1234)

# Manejar el desbalance de clases utilizando SMOTE
smote = SMOTE(random_state=1234)
X_train_resampled, Y_train_resampled = smote.fit_resample(X_train, Y_train)

# Estandarizar las características
scaler = StandardScaler()
X_train_resampled_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)

# Modelo: XGBoost con ajuste de hiperparámetros
xgb = XGBClassifier(random_state=1234, eval_metric='logloss') 

xgb_params = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1],
    'max_depth': [3, 5],
}

xgb_grid = GridSearchCV(xgb, xgb_params, scoring='f1', cv=3)
xgb_grid.fit(X_train_resampled_scaled, Y_train_resampled)

# Predicciones del modelo XGBoost
Y_hat_test_xgb = xgb_grid.best_estimator_.predict(X_test_scaled)
roc_auc_test_xgb = roc_auc_score(Y_test, xgb_grid.best_estimator_.predict_proba(X_test_scaled)[:, 1])

# Métricas de evaluación para el modelo XGBoost
metrics_xgb = {
    'Accuracy': accuracy_score(Y_test, Y_hat_test_xgb),
    'F1': f1_score(Y_test, Y_hat_test_xgb),
    'Precision': precision_score(Y_test, Y_hat_test_xgb),
    'Recall': recall_score(Y_test, Y_hat_test_xgb),
    'ROC-AUC': roc_auc_test_xgb
}

# Imprimir resultados del modelo XGBoost
print("Desempeño de XGBoost:")
print(metrics_xgb)

# Guardar el mejor modelo (XGBoost)
with open("./models/XGBoost_model.pkl", "wb") as file:
    pickle.dump(xgb_grid.best_estimator_, file)

# Guardar el StandardScaler utilizado en el entrenamiento
with open("./models/xgb_scaler.pkl", "wb") as scaler_file:
    pickle.dump(scaler, scaler_file)
