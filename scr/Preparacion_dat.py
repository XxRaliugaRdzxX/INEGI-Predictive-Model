import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('../data/dataset_consolidado.csv')

df_clean = df[df['UNI_VEH'] > 0].copy()

# Variables de entrada (X)
feature_cols = ['ANIO', 'ID_MES', 'MARCA', 'TIPO', 'SEGMENTO', 'ORIGEN', 'ID_PAIS_ORIGEN']
X_raw = df_clean[feature_cols]
y = df_clean['UNI_VEH'] # Variable objetivo (Y)

# Codificación de variables categóricas a dummies
X = pd.get_dummies(X_raw, columns=['MARCA', 'TIPO', 'SEGMENTO', 'ORIGEN'], drop_first=True)

print(f"Dimensiones de X (Variables de entrada): {X.shape}")
print(f"Dimensiones de y (Variable objetivo): {y.shape}")

# FASE 3: División del Dataset (Train / Test Split)
# Separamos 80% de los datos para entrenamiento y 20% para la prueba final
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\n Divison completa")
print(f"Registros para Entrenamiento (X_train): {X_train.shape[0]}")
print(f"Registros para Prueba (X_test): {X_test.shape[0]}")