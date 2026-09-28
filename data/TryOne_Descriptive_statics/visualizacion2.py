import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar el dataset (asegúrate de tener 'dataset_automotriz_limpio.csv' en tu directorio)
df = pd.read_csv("dataset_automotriz_limpio.csv")

# Preparar los datos
df_anio = df.groupby('ANIO')['UNI_VEH'].sum().reset_index()
df_anio['UNI_VEH_MILLONES'] = df_anio['UNI_VEH'] / 1_000_000

# Configurar estilo y figura
sns.set_theme(style="whitegrid")
plt.figure(figsize=(14, 7))

# Graficar la línea principal
sns.lineplot(
    data=df_anio, x='ANIO', y='UNI_VEH_MILLONES', 
    marker='o', linewidth=2.5, color='#2c3e50', zorder=4
)

# ==========================================================
# AGREGAR CONTEXTO MACROECONÓMICO (STORYTELLING)
# ==========================================================

# 1. Crisis Global (2008-2009): Sombreado rojo
plt.axvspan(2008, 2009, color='#e74c3c', alpha=0.15, zorder=1)
plt.text(2008.5, 0.4, 'Crisis Global\n(2008-2009)', 
         horizontalalignment='center', color='#c0392b', fontweight='bold')

# 2. Pico Histórico (2016): Flecha de anotación
pico_x = 2016
pico_y = df_anio[df_anio['ANIO'] == 2016]['UNI_VEH_MILLONES'].values[0]
plt.annotate(
    'Pico Histórico\n(> 1.6M)', 
    xy=(pico_x, pico_y), xytext=(pico_x, pico_y + 0.15),
    arrowprops=dict(facecolor='#27ae60', shrink=0.05, width=2, headwidth=8),
    horizontalalignment='center', color='#27ae60', fontweight='bold'
)

# 3. Caída por Pandemia (2020): Flecha de anotación
pandemia_x = 2020
pandemia_y = df_anio[df_anio['ANIO'] == 2020]['UNI_VEH_MILLONES'].values[0]
plt.annotate(
    'Desplome por\nPandemia (2020)', 
    xy=(pandemia_x, pandemia_y), xytext=(pandemia_x, pandemia_y - 0.25),
    arrowprops=dict(facecolor='#c0392b', shrink=0.05, width=2, headwidth=8),
    horizontalalignment='center', color='#c0392b', fontweight='bold'
)

# 4. Recuperación (2021-2025): Sombreado verde
plt.axvspan(2021, 2025, color='#2ecc71', alpha=0.15, zorder=1)
plt.text(2023, 0.4, 'Recuperación\nPost-Pandemia', 
         horizontalalignment='center', color='#27ae60', fontweight='bold')

# 5. Año Incompleto (2026): Sombreado gris
plt.axvspan(2025.5, 2026.5, color='grey', alpha=0.2, zorder=1)
plt.text(2026, 0.4, 'Falso Desplome\n(Datos 2026\nincompletos)', 
         horizontalalignment='center', color='#7f8c8d', fontweight='bold', fontsize=9)

# ==========================================================
# FORMATO FINAL DE LA GRÁFICA
# ==========================================================

plt.title('Evolución Histórica y Eventos Macroeconómicos en la Industria Automotriz', 
          fontsize=16, pad=20, fontweight='bold')
plt.xlabel('Año', fontsize=12, fontweight='bold')
plt.ylabel('Total de Unidades Vendidas (Millones)', fontsize=12, fontweight='bold')

# Ajustar límites del eje Y para hacer espacio para los textos
plt.ylim(0.2, df_anio['UNI_VEH_MILLONES'].max() + 0.3)

# Asegurar que todos los años aparezcan en el eje X
plt.xticks(df_anio['ANIO'], rotation=45)

plt.tight_layout()
plt.show()