import pandas as pd


def auditar_por_columna(df: pd.DataFrame) -> None:
    print("=" * 70)
    print("        AUDITORÍA Y MUESTREO DETALLADO POR COLUMNA")
    print("=" * 70)

    total_filas = len(df)
    print(f"Total de registros a evaluar: {total_filas:,}\n")

    for col in df.columns:
        print(f"COLUMNA: '{col}'")
        print("-" * 50)

        # 1. Tipo de dato y nulos
        dtype = df[col].dtype
        nulos = df[col].isnull().sum()
        pct_nulos = (nulos / total_filas) * 100

        # 2. Análisis de únicos y duplicados a nivel columna
        unicos = df[col].nunique(dropna=False)
        duplicados_col = total_filas - unicos

        print(f"  • Tipo de dato:          {dtype}")
        print(f"  • Valores nulos:          {nulos:,} ({pct_nulos:.2f}%)")
        print(f"  • Valores únicos:         {unicos:,}")
        print(f"  • Repeticiones/Frecuencia:{duplicados_col:,}")

        # 3. Muestra de los 5 valores más frecuentes
        print("\n  Top 5 valores más frecuentes en esta columna:")
        top_valores = df[col].value_counts(dropna=False).head(5)
        for valor, val_count in top_valores.items():
            pct_val = (val_count / total_filas) * 100
            print(f"    - [{valor}]: {val_count:,} apariciones ({pct_val:.2f}%)")

        print("=" * 70)


def limpiar_dataset(
    df: pd.DataFrame, columnas_a_borrar: list
) -> pd.DataFrame:
    """Elimina columnas constantes/innecesarias y retorna una copia limpia del DataFrame."""
    print(f"\n[+] Eliminando columnas sin varianza: {columnas_a_borrar}...")
    df_resultado = df.drop(columns=columnas_a_borrar)
    print(f"✔ Columnas resultantes ({len(df_resultado.columns)}):")
    print(list(df_resultado.columns))
    return df_resultado


# --- EJECUCIÓN DEL PIPELINE ---
if __name__ == "__main__":
    # 1. Cargar el dataset consolidado
    df = pd.read_csv("dataset_consolidado.csv")

    # 2. Ejecutar auditoría inicial
    auditar_por_columna(df)

    # 3. Aplicar limpieza/eliminación de columnas constantes
    columnas_inutiles = ["PROD_EST", "COBERTURA"]
    df_limpio = limpiar_dataset(df, columnas_a_borrar=columnas_inutiles)

    # 4. (Opcional) Guardar el dataset listo para visualización
    df_limpio.to_csv("dataset_automotriz_limpio.csv", index=False)
    print(
        "\n¡Proceso de limpieza completado! Guardado en 'dataset_automotriz_limpio.csv'."
    )