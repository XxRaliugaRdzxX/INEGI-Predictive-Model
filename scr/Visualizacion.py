import pandas as pd

df = pd.read_csv('../data/dataset_consolidado.csv')

print(df.head(5))
df.info()
print(df.describe())

df_clean = df[df['UNI_VEH'] > 0].copy()

print("--- Valores únicos por columna ---")
for col in df_clean.columns:
    print(f"{col}: {df_clean[col].nunique()} valores únicos")


print(f"\nFilas originales: {len(df)}")
print(f"Filas tras eliminar unidades <= 0: {len(df_clean)}")