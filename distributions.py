import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp

# Cargar el dataset
file_path = './data/credit_train.csv'
data = pd.read_csv(file_path)

# Separar variables categóricas y continuas
categorical_cols = ['X2', 'X3', 'X4', 'X6', 'X7', 'X8', 'X9', 'X10', 'X11']  # Categóricas según análisis previo
continuous_cols = ['X1', 'X5', 'X12', 'X13', 'X14', 'X15', 'X16', 'X17', 'X18', 'X19', 'X20', 'X21', 'X22', 'X23']  # Continuas según análisis previo

# -------------------------
# Prueba de Chi-Cuadrado para variables categóricas
# -------------------------
chi2_results = {}
for col in categorical_cols:
    contingency_table = pd.crosstab(data[col], data['Y'])  # Y es la variable dependiente
    chi2_stat, p_val, _, _ = chi2_contingency(contingency_table)
    chi2_results[col] = p_val  # Guardamos el valor p para cada variable categórica

# -------------------------
# Prueba de Kolmogorov-Smirnov para variables continuas
# -------------------------
ks_results = {}
for col in continuous_cols:
    class_0 = data[data['Y'] == 0][col]  # Datos donde Y = 0
    class_1 = data[data['Y'] == 1][col]  # Datos donde Y = 1
    ks_stat, p_val = ks_2samp(class_0, class_1)
    ks_results[col] = p_val  # Guardamos el valor p para cada variable continua

# Mostramos los resultados
print("Resultados Chi-Cuadrado (variables categóricas):")
for col, p_val in chi2_results.items():
    print(f"Variable {col}: Valor p = {p_val}")

print("\nResultados Kolmogorov-Smirnov (variables continuas):")
for col, p_val in ks_results.items():
    print(f"Variable {col}: Valor p = {p_val}")
