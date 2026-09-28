import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar el dataset limpio
df = pd.read_csv("dataset_automotriz_limpio.csv")

# Configurar el estilo visual global de Seaborn
sns.set_theme(style="whitegrid", palette="muted")

# ==============================================================================
# 1. TENDENCIA ANUAL DE UNIDADES
# ==============================================================================
plt.figure(figsize=(12, 5))
# Agrupamos por año sumando las unidades
df_anio = df.groupby('ANIO')['UNI_VEH'].sum().reset_index()

sns.lineplot(data=df_anio, x='ANIO', y='UNI_VEH', marker='o', linewidth=2.5, color='#2c3e50')
plt.title('Evolución Histórica: Unidades de Vehículos por Año', fontsize=14, pad=15)
plt.xlabel('Año', fontsize=12)
plt.ylabel('Total de Unidades', fontsize=12)
plt.xticks(df_anio['ANIO'], rotation=45)
# Rellenar el área bajo la curva para darle mejor estética
plt.fill_between(df_anio['ANIO'], df_anio['UNI_VEH'], alpha=0.2, color='#2c3e50')
plt.tight_layout()
plt.show()

# ==============================================================================
# 2. TOP 10 MARCAS MÁS VENDIDAS
# ==============================================================================
plt.figure(figsize=(10, 6))
# Agrupamos por marca, sumamos unidades y tomamos el Top 10
df_marcas = df.groupby('MARCA')['UNI_VEH'].sum().reset_index().sort_values('UNI_VEH', ascending=False).head(10)

sns.barplot(data=df_marcas, x='UNI_VEH', y='MARCA', hue='MARCA', palette='viridis', legend=False)
plt.title('Top 10 Marcas Automotrices por Volumen de Unidades', fontsize=14, pad=15)
plt.xlabel('Total de Unidades', fontsize=12)
plt.ylabel('Marca', fontsize=12)
plt.tight_layout()
plt.show()

# ==============================================================================
# 3. PREFERENCIA POR SEGMENTO Y TIPO
# ==============================================================================
plt.figure(figsize=(12, 6))
# Agrupamos por segmento y sumamos
df_segmento = df.groupby(['SEGMENTO', 'TIPO'])['UNI_VEH'].sum().reset_index().sort_values('UNI_VEH', ascending=False)

sns.barplot(data=df_segmento, x='UNI_VEH', y='SEGMENTO', hue='TIPO', palette='Set2')
plt.title('Volumen de Unidades por Segmento y Tipo de Vehículo', fontsize=14, pad=15)
plt.xlabel('Total de Unidades', fontsize=12)
plt.ylabel('Segmento', fontsize=12)
plt.legend(title='Tipo de Vehículo')
plt.tight_layout()
plt.show()

# ==============================================================================
# 4. PROPORCIÓN: IMPORTADO VS NACIONAL
# ==============================================================================
plt.figure(figsize=(7, 7))
df_origen = df.groupby('ORIGEN')['UNI_VEH'].sum().reset_index()

colores = ['#ff9999','#66b3ff']
explode = (0.05, 0) # Separar un poco la primera rebanada

plt.pie(df_origen['UNI_VEH'], labels=df_origen['ORIGEN'], autopct='%1.1f%%', 
        startangle=140, colors=colores, explode=explode, shadow=True, textprops={'fontsize': 12})
plt.title('Proporción del Volumen de Unidades: Nacional vs Importado', fontsize=14, pad=15)
plt.tight_layout()
plt.show()