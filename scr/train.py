import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

df = pd.read_csv('../data/dataset_consolidado.csv')
df_clean = df[df['UNI_VEH'] > 0].copy()

feature_cols = ['ANIO', 'ID_MES', 'MARCA', 'TIPO', 'SEGMENTO', 'ORIGEN', 'ID_PAIS_ORIGEN']
X_raw = df_clean[feature_cols]
y = df_clean['UNI_VEH']

X = pd.get_dummies(X_raw, columns=['MARCA', 'TIPO', 'SEGMENTO', 'ORIGEN'], drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# FASE 4: Entrenamiento de los Modelos

print("Entrenando Regresión Lineal...")
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

print("Entrenando Random Forest Regressor (esto puede demorar unos segundos)...")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

# FASE 5: Evaluación del Modelo
def evaluar_modelo(nombre, modelo, X_test, y_test):
    predictions = modelo.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    
    print(f"\n--- Resultados: {nombre} ---")
    print(f"MAE  (Error Absoluto Medio): {mae:.2f} unidades")
    print(f"RMSE (Raíz de Error Cuadrático Medio): {rmse:.2f} unidades")
    print(f"R²   (Coeficiente de Determinación): {r2:.4f}")

evaluar_modelo("Regresión Lineal", lr_model, X_test, y_test)
evaluar_modelo("Random Forest Regressor", rf_model, X_test, y_test)

# =============================================================
# FASE 6: Predicción Final con Nuevos Datos
# =============================================================

# Tomamos una fila de ejemplo del conjunto de prueba para simular una consulta nueva
ejemplo_x = X_test.iloc[[0]]
venta_real = y_test.iloc[0]

# Realizamos la predicción con nuestro mejor modelo (Random Forest)
prediccion_rf = rf_model.predict(ejemplo_x)[0]

print("\n--- Demostración de Predicción en Tiempo Real ---")
print(f"Venta Real Registrada:   {venta_real} unidades")
print(f"Predicción del Modelo:   {prediccion_rf:.1f} unidades")
print(f"Diferencia / Error:      {abs(venta_real - prediccion_rf):.1f} unidades")

# =============================================================
# FASE 7: Gráfica de Dispersión de Regresión Lineal
# =============================================================

y_pred_lr = lr_model.predict(X_test)

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_lr, alpha=0.5, color='blue', label='Predicciones')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Ideal (y = x)')
plt.xlabel('Valores Reales')
plt.ylabel('Valores Predichos')
plt.title('Regresión Lineal: Reales vs Predichos')
plt.legend()
plt.show()