import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =============================================================
# 1. Carga y NUEVO FILTRADO TEMPORAL
# =============================================================
df = pd.read_csv('../data/dataset_consolidado.csv')

# Filtramos ventas válidas y acotamos a la época de estabilización (2022 en adelante)
df_clean = df[(df['UNI_VEH'] > 0) & (df['ANIO'] >= 2022)].copy()

print(f"Filas tras aplicar filtro temporal (2022-2026): {len(df_clean)}")

# =============================================================
# 2. Preprocesamiento
# =============================================================
feature_cols = ['ANIO', 'ID_MES', 'MARCA', 'TIPO', 'SEGMENTO', 'ORIGEN', 'ID_PAIS_ORIGEN']
X_raw = df_clean[feature_cols]
y = df_clean['UNI_VEH']

# Codificación y división
X = pd.get_dummies(X_raw, columns=['MARCA', 'TIPO', 'SEGMENTO', 'ORIGEN'], drop_first=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# =============================================================
# 3. Entrenamiento (Solo Random Forest)
# =============================================================
print("Entrenando Random Forest con datos estabilizados...")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

# Evaluación
predicciones = rf_model.predict(X_test)
r2 = r2_score(y_test, predicciones)
mae = mean_absolute_error(y_test, predicciones)

print("\n--- Resultados con Datos Estabilizados ---")
print(f"R²   (Coeficiente de Determinación): {r2:.4f}")
print(f"MAE  (Error Absoluto Medio): {mae:.2f} unidades")

# =============================================================
# 4. Visualización de Resultados
# =============================================================
# Configuración visual
plt.figure(figsize=(10, 6))
sns.set_theme(style="whitegrid")

# Crear gráfica de dispersión: Real vs Predicción
plt.scatter(y_test, predicciones, alpha=0.5, color='blue', edgecolor='k')

# Trazar la línea ideal (donde Real = Predicción)
limite_max = max(max(y_test), max(predicciones))
plt.plot([0, limite_max], [0, limite_max], color='red', linestyle='--', linewidth=2, label='Predicción Perfecta')

plt.title(f'Ventas Reales vs Predicciones (R² = {r2:.2f})', fontsize=14)
plt.xlabel('Ventas Reales (Unidades)', fontsize=12)
plt.ylabel('Predicción del Modelo (Unidades)', fontsize=12)
plt.legend()
plt.tight_layout()

# Mostrar la gráfica en una ventana emergente
plt.show()