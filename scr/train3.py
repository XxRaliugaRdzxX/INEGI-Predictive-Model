import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Carga y limpieza
df = pd.read_csv('../data/dataset_consolidado.csv')
df_clean = df[df['UNI_VEH'] > 0].copy()

# 2. Selección estricta de variables (Tu propuesta)
feature_cols = ['ANIO', 'MARCA', 'SEGMENTO']
X_raw = df_clean[feature_cols]
y = df_clean['UNI_VEH']

# IMPORTANTE: Aplicamos get_dummies antes de separar por años para asegurar 
# que Train y Test tengan exactamente la misma cantidad de columnas.
X_encoded = pd.get_dummies(X_raw, columns=['MARCA', 'SEGMENTO'], drop_first=True)

# 3. División Temporal (Train: 2021-2024 | Test: 2026)
mask_train = df_clean['ANIO'].isin([2021, 2022, 2023, 2024])
mask_test = df_clean['ANIO'] == 2026

X_train = X_encoded[mask_train]
y_train = y[mask_train]

X_test = X_encoded[mask_test]
y_test = y[mask_test]

print(f"--- Dimensiones del Experimento ---")
print(f"Entrenamiento (2021-2024): {len(X_train)} registros")
print(f"Prueba (2026):             {len(X_test)} registros\n")

# 4. Entrenamiento de Modelos
print("Entrenando Regresión Lineal...")
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

print("Entrenando Random Forest...")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

# 5. Evaluación 2026
pred_lr = lr_model.predict(X_test)
pred_rf = rf_model.predict(X_test)

print("--- Resultados Predicción 2026 ---")
print(f"Regresión Lineal -> R²: {r2_score(y_test, pred_lr):.4f} | MAE: {mean_absolute_error(y_test, pred_lr):.2f}")
print(f"Random Forest    -> R²: {r2_score(y_test, pred_rf):.4f} | MAE: {mean_absolute_error(y_test, pred_rf):.2f}")

# 6. Gráfica Comparativa (Usando el mejor modelo, en este caso graficaremos ambos para comparar)
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(y_test, pred_lr, alpha=0.5, color='orange', edgecolor='k')
limite = max(max(y_test), max(pred_lr))
plt.plot([0, limite], [0, limite], 'r--', lw=2)
plt.title(f'Regresión Lineal 2026 (R²={r2_score(y_test, pred_lr):.2f})')
plt.xlabel('Ventas Reales 2026')
plt.ylabel('Predicción')

plt.subplot(1, 2, 2)
plt.scatter(y_test, pred_rf, alpha=0.5, color='blue', edgecolor='k')
limite = max(max(y_test), max(pred_rf))
plt.plot([0, limite], [0, limite], 'r--', lw=2)
plt.title(f'Random Forest 2026 (R²={r2_score(y_test, pred_rf):.2f})')
plt.xlabel('Ventas Reales 2026')

plt.tight_layout()
plt.show()