import glob
import pandas as pd

# Directorio de descarga de los csv
archivos_csv = glob.glob("/home/rali36/Downloads/conjunto_dataset/conjunto_de_datos/*.csv")

print(f"Se encontraron {len(archivos_csv)} archivos CSV para unir.")

# Se Lee cada CSV y se guarda en una lista de DataFrames
lista_dfs = [pd.read_csv(archivo) for archivo in archivos_csv]

# Se Concatena todos los DataFrames en uno solo
df_unificado = pd.concat(lista_dfs, ignore_index=True)

# Se guarda el resultado en un nuevo archivo CSV
df_unificado.to_csv("dataset_consolidado.csv", index=False)
