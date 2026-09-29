import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Librerías para visualización
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import plot_tree

# FASE 1 - 3: Carga y Preparación de Datos
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
    return predictions

y_pred_lr = evaluar_modelo("Regresión Lineal", lr_model, X_test, y_test)
y_pred_rf = evaluar_modelo("Random Forest Regressor", rf_model, X_test, y_test)

# FASE 6: Predicción Final con Nuevos Datos
ejemplo_x = X_test.iloc[[0]]
venta_real = y_test.iloc[0]
prediccion_rf = rf_model.predict(ejemplo_x)[0]

print("\n--- Demostración de Predicción en Tiempo Real ---")
print(f"Venta Real Registrada:   {venta_real} unidades")
print(f"Predicción del Modelo:   {prediccion_rf:.1f} unidades")
print(f"Diferencia / Error:      {abs(venta_real - prediccion_rf):.1f} unidades")


# =============================================================
# FASE 7: VISUALIZACIONES Y GRÁFICAS DE EVALUACIÓN
# =============================================================

sns.set_theme(style="whitegrid")

# -------------------------------------------------------------
# 1. EVALUACIÓN DE DISPERSIÓN EN REGRESIÓN LINEAL
# -------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# 1.1 Gráfico de Dispersión: Real vs Predicho
axes[0].scatter(y_test, y_pred_lr, alpha=0.4, color='royalblue', edgecolors='k', linewidth=0.5)
max_val = max(y_test.max(), y_pred_lr.max())
axes[0].plot([0, max_val], [0, max_val], 'r--', lw=2, label='Predicción Perfecta ($y = x$)')
axes[0].set_xlabel('Ventas Reales (`UNI_VEH`)')
axes[0].set_ylabel('Ventas Predichas por Regresión Lineal')
axes[0].set_title('Regresión Lineal: Dispersión Real vs. Predicción')
axes[0].legend()

# 1.2 Gráfico de Residuos (Errores)
residuos_lr = y_test - y_pred_lr
axes[1].scatter(y_pred_lr, residuos_lr, alpha=0.4, color='purple', edgecolors='k', linewidth=0.5)
axes[1].axhline(y=0, color='red', linestyle='--', lw=2)
axes[1].set_xlabel('Valores Predichos ($\hat{y}$)')
axes[1].set_ylabel('Residuos / Error ($y - \hat{y}$)')
axes[1].set_title('Regresión Lineal: Análisis de Residuos')

plt.tight_layout()
plt.show()


# -------------------------------------------------------------
# 2. EVALUACIÓN Y GRÁFICAS DE RANDOM FOREST
# -------------------------------------------------------------

# 2.1 Importancia de las Top 15 Variables en el Random Forest
importancias = rf_model.feature_importances_
nombres_variables = X_train.columns
indices = np.argsort(importancias)[::-1][:15]  # Selecciona las 15 más importantes

plt.figure(figsize=(10, 6))
plt.title('Random Forest: Top 15 Variables Más Importantes')
plt.barh(range(len(indices)), importancias[indices][::-1], color='forestgreen', align='center')
plt.yticks(range(len(indices)), [nombres_variables[i] for i in indices][::-1])
plt.xlabel('Importancia Relativa (Gini Importance)')
plt.tight_layout()
plt.show()

# 2.2 Estructura de 1 Árbol de Decisión del Bosque
plt.figure(figsize=(18, 8))
plot_tree(
    rf_model.estimators_[0],
    feature_names=X_train.columns,
    filled=True,
    max_depth=3,        # Profundidad limitada para mantener legibilidad
    fontsize=9
)
plt.title('Random Forest: Estructura de 1 Árbol de Decisión Individual (Primeros 3 Niveles)')
plt.tight_layout()
plt.show()