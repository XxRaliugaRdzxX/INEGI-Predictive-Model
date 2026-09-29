# Practica de Manejo masivo de datos [Titanic_Bigdata_KDAR]

> A lo largo de esta practica se trata de realizar un modelo de entrenamiento sencillo, con el fin de poner en práctica una predicción del número de unidades vendidas de vehiculos ligero en México, este suceso aún no sucede. Sin embargo, gracias a esta disciplina podemos inducir un valor.

---

## DataSet 
- dataset_consolidado.csv(22Mb)
- https://
- Datos sobre ventas, modelos, categorias de carros desde el 2025.
---

## Objetivo
- Qué número de unidades de vehiculos ligeros serán vendidas en 2026
---

## Requisitos 
- Se requiere Python y de ser posible una distribución de Linux
- Dependencias descargadas del archivo Requirements.txt en un entorno
---

## Instalación

Sigue estos pasos para ejecutar el proyecto en tu entorno local:

### 1. Clonar el repositorio
git clone https://github.com/XxRaliugaRdzxX/INEGI-Predictive-Model.git

### 2. Entrar al directorio del proyecto
cd INEGI-Predictive-Model

### 3. Instalar las dependencias
- Primero el entorno con el comando `python -m venv .venv`
- Despues activar el entorno `source .venv/bin/activate`
- Al final descargar las dependecias con `pip install -r requirements.txt`


### 4. Iniciar el proyecto
- Correr el comando del script del análisis con `python src/analysis.py`

## 5. Conclusiones 
Variable Objetivo(Y): UNI_VEH
Variables de Entrada(X): 
- TEMPORARES --> ANIO, ID_MES
- CATEGORIAS --> MARCA, MODELO, TIPO, SEGMENTO 
- PROCEDENCIA --> ORIGEN, ID_PAIS_ORIGEN

Columnas eliminadas por su irrelevancia:
PROD_EST y COBERTURA tienen 1 solo valor único. Al ser constantes en todo el dataset, no aportan variabilidad ni poder predictivo al modelo.   

ESTATUS solo tiene 2 valores únicos ("Cifras Definitivas" | "Cifras Preliminares"), al ser un metadato administrativo no afecta el volumen real de ventas.


Al final se llego a la conclusión de que el mejor metodo para predecir bajo la naturaleza de este data set fue regresión lineal. No obstante, el modelo no puede ocupar la información de todo el set debido a que a lo largo del 2008 y 2021 ocurrieron fenomenos que mueven las metricas de manera abismal.

Por lo que solamente se ocuparon dato del 2022 al 2025, dando un margen de explicación del 50% de los datos en la regresión lineal.

Desarrollado con ❤️ por [XxRaliugaRdzxX](github.com/XxRaliugaRdzxX/)
