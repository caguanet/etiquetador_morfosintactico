# Etiquetador Morfosintáctico

Este proyecto implementa un etiquetador morfosintáctico utilizando el algoritmo de Viterbi para el análisis de texto en español. El sistema es capaz de asignar etiquetas morfosintácticas (POS tags) a palabras en una oración, basándose en probabilidades de emisión y transición calculadas a partir de un corpus de entrenamiento.

## Características Principales

- Implementación del algoritmo de Viterbi para encontrar la mejor secuencia de etiquetas
- Cálculo de probabilidades de emisión y transición con suavizado
- Análisis de resultados y estadísticas del corpus
- Extracción de características morfológicas
- Exportación de resultados a archivos Excel

## Requisitos

- Python 3.x
- NumPy
- Pandas

## Estructura del Proyecto

El proyecto consta de los siguientes componentes principales:

- `etiquetador_morfosintactico.py`: Módulo principal que contiene toda la lógica del etiquetador
- `corpus_etiquetado.txt`: Archivo de entrada que debe contener el corpus etiquetado (no incluido)
- `resultados/`: Directorio donde se guardan los archivos de salida
  - `probabilidades_emision.xlsx`
  - `probabilidades_transicion.xlsx`
  - `matriz_viterbi.xlsx`

## Funciones Principales

1. `recalcular_probabilidades_normalizadas()`: Calcula las probabilidades de emisión y transición
2. `viterbi()`: Implementa el algoritmo de Viterbi para la etiquetación
3. `guardar_resultados()`: Exporta los resultados a archivos Excel
4. `analizar_resultados()`: Muestra los resultados de la etiquetación
5. `estadisticas_corpus()`: Genera estadísticas sobre el corpus utilizado

## Uso

1. Preparar un archivo de corpus etiquetado en formato de texto
2. Ejecutar el script principal:
```python
python etiquetador_morfosintactico.py
```

## Formato del Corpus

El corpus debe estar en formato de texto con el siguiente formato:
```
palabra    lema    etiqueta
```

## Salida

El sistema genera tres archivos Excel en el directorio `resultados/`:
- Matriz de probabilidades de emisión
- Matriz de probabilidades de transición
- Matriz de Viterbi para la última oración procesada

## Características Adicionales

- Suavizado de probabilidades para manejar palabras desconocidas
- Extracción de características morfológicas (sufijos, prefijos, capitalización)
- Soporte para análisis de trigramas 