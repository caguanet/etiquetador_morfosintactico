"""
Este módulo implementa un etiquetador morfosintáctico utilizando el algoritmo de Viterbi.
Se encarga de calcular probabilidades de emisión y transición, así como de analizar resultados.

Funciones:
- recalcular_probabilidades_normalizadas: Calcula las probabilidades de emisión y transición a partir de un corpus.
- viterbi: Aplica el algoritmo de Viterbi para encontrar la mejor secuencia de etiquetas para una oración dada.
- guardar_resultados: Guarda las probabilidades y la matriz de Viterbi en archivos Excel.
- cargar_corpus_desde_archivo: Carga un corpus etiquetado desde un archivo.
- analizar_resultados: Analiza y muestra los resultados de la etiquetación.
- estadisticas_corpus: Imprime estadísticas sobre el corpus, como tamaño y vocabulario.
- calcular_probabilidades_trigrama: Calcula probabilidades de transición en un modelo de trigrama.
- extraer_caracteristicas_morfologicas: Extrae características morfológicas de una palabra.
"""

import numpy as np
import pandas as pd
from collections import defaultdict
from typing import List, Tuple, Dict, Set
import os

def recalcular_probabilidades_normalizadas(corpus: List[Tuple[str, str]], suavizado: float = 1e-6) -> Tuple[Dict, Dict, Set]:
    """
    Calcula las probabilidades de emisión y transición a partir de un corpus.

    Parámetros:
    - corpus: Lista de tuplas (token, etiqueta).
    - suavizado: Valor de suavizado para evitar probabilidades cero.

    Retorna:
    - prob_emision: Diccionario de probabilidades de emisión.
    - prob_transicion: Diccionario de probabilidades de transición.
    - etiquetas: Conjunto de etiquetas únicas.
    """
    emisiones = defaultdict(lambda: defaultdict(float))
    transiciones = defaultdict(lambda: defaultdict(float))
    etiquetas = set()

    prev_etiqueta = "<START>"
    for token, etiqueta in corpus:
        emisiones[etiqueta][token] += 1
        transiciones[prev_etiqueta][etiqueta] += 1
        etiquetas.add(etiqueta)
        prev_etiqueta = etiqueta

    prob_emision = defaultdict(lambda: defaultdict(lambda: suavizado))
    prob_transicion = defaultdict(lambda: defaultdict(lambda: suavizado))

    for etiqueta, tokens in emisiones.items():
        total = sum(tokens.values()) + suavizado * len(tokens)
        for token, freq in tokens.items():
            prob_emision[etiqueta][token] = (freq + suavizado) / total

    for prev, siguiente in transiciones.items():
        total = sum(siguiente.values()) + suavizado * len(siguiente)
        for curr, freq in siguiente.items():
            prob_transicion[prev][curr] = (freq + suavizado) / total

    return prob_emision, prob_transicion, etiquetas

def viterbi(oracion: List[str], etiquetas: Set[str], prob_emision: Dict, prob_transicion: Dict) -> Tuple[List[str], np.ndarray]:
    """
    Aplica el algoritmo de Viterbi para encontrar la mejor secuencia de etiquetas.

    Parámetros:
    - oracion: Lista de palabras en la oración.
    - etiquetas: Conjunto de etiquetas posibles.
    - prob_emision: Diccionario de probabilidades de emisión.
    - prob_transicion: Diccionario de probabilidades de transición.

    Retorna:
    - mejor_secuencia: Lista de etiquetas predichas.
    - matriz_viterbi: Matriz de probabilidades de Viterbi.
    """
    n = len(oracion)
    etiqueta_idx = list(etiquetas)
    matriz_viterbi = np.zeros((len(etiquetas), n))
    matriz_traza = np.zeros((len(etiquetas), n), dtype=int)

    for i, etiqueta in enumerate(etiqueta_idx):
        matriz_viterbi[i, 0] = prob_transicion["<START>"].get(etiqueta, 1e-12) * prob_emision[etiqueta].get(oracion[0], 1e-12)

    for t in range(1, n):
        for i, etiqueta in enumerate(etiqueta_idx):
            prob_emisiones = prob_emision[etiqueta].get(oracion[t], 1e-12)
            probabilidades = [
                matriz_viterbi[j, t-1] * prob_transicion[etiqueta_idx[j]].get(etiqueta, 1e-12) * prob_emisiones
                for j in range(len(etiqueta_idx))
            ]
            matriz_viterbi[i, t] = max(probabilidades)
            matriz_traza[i, t] = np.argmax(probabilidades)

    mejor_secuencia = []
    idx_actual = np.argmax(matriz_viterbi[:, -1])
    for t in range(n-1, -1, -1):
        mejor_secuencia.insert(0, etiqueta_idx[idx_actual])
        idx_actual = matriz_traza[idx_actual, t]

    return mejor_secuencia, matriz_viterbi

def guardar_resultados(prob_emision, prob_transicion, matriz_viterbi, oracion, etiquetas):
    """
    Guarda las probabilidades y la matriz de Viterbi en archivos Excel.

    Parámetros:
    - prob_emision: Diccionario de probabilidades de emisión.
    - prob_transicion: Diccionario de probabilidades de transición.
    - matriz_viterbi: Matriz de probabilidades de Viterbi.
    - oracion: Lista de palabras en la oración.
    - etiquetas: Conjunto de etiquetas únicas.
    """
    os.makedirs("resultados", exist_ok=True)
    
    df_emision = pd.DataFrame(prob_emision).fillna(0)
    df_emision.to_excel("resultados/probabilidades_emision.xlsx")
    
    df_transicion = pd.DataFrame(prob_transicion).fillna(0)
    df_transicion.to_excel("resultados/probabilidades_transicion.xlsx")
    
    df_viterbi = pd.DataFrame(
        matriz_viterbi,
        index=list(etiquetas),
        columns=oracion
    )
    df_viterbi.to_excel("resultados/matriz_viterbi.xlsx")

def cargar_corpus_desde_archivo(ruta: str) -> List[Tuple[str, str]]:
    """
    Carga un corpus etiquetado desde un archivo.

    Parámetros:
    - ruta: Ruta del archivo que contiene el corpus.

    Retorna:
    - corpus: Lista de tuplas (token, etiqueta).
    """
    corpus = []
    with open(ruta, "r", encoding="utf-8") as archivo:
        for linea in archivo:
            if linea.strip() and not linea.startswith("<doc"):
                partes = linea.strip().split("\t")
                if len(partes) >= 3:
                    token, _, etiqueta = partes[:3]
                    corpus.append((token.lower(), etiqueta))
    return corpus

def analizar_resultados(oracion, etiquetas_predichas, etiquetas_gold=None):
    """
    Analiza y muestra los resultados de la etiquetación.

    Parámetros:
    - oracion: Lista de palabras en la oración.
    - etiquetas_predichas: Lista de etiquetas predichas.
    - etiquetas_gold: Lista de etiquetas reales (opcional).
    """
    print("\nAnálisis de Resultados:")
    print("-" * 50)
    for palabra, etiqueta in zip(oracion, etiquetas_predichas):
        print(f"{palabra}: {etiqueta}")
    
    if etiquetas_gold:
        precision = sum(p == g for p, g in zip(etiquetas_predichas, etiquetas_gold)) / len(etiquetas_predichas)
        print(f"\nPrecisión: {precision:.2%}")

def estadisticas_corpus(corpus):
    """
    Imprime estadísticas sobre el corpus.

    Parámetros:
    - corpus: Lista de tuplas (token, etiqueta).
    """
    vocabulario = set(token for token, _ in corpus)
    etiquetas = set(etiqueta for _, etiqueta in corpus)
    print(f"Tamaño del corpus: {len(corpus)}")
    print(f"Tamaño del vocabulario: {len(vocabulario)}")
    print(f"Número de etiquetas únicas: {len(etiquetas)}")

def calcular_probabilidades_trigrama(corpus):
    """
    Calcula probabilidades de transición en un modelo de trigrama.

    Parámetros:
    - corpus: Lista de tuplas (token, etiqueta).
    """
    emisiones = defaultdict(lambda: defaultdict(float))
    transiciones = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    
    prev_prev = "<START>"
    prev = "<START>"
    for token, etiqueta in corpus:
        emisiones[etiqueta][token] += 1
        transiciones[prev_prev][prev][etiqueta] += 1
        prev_prev = prev
        prev = etiqueta

def extraer_caracteristicas_morfologicas(palabra):
    """
    Extrae características morfológicas de una palabra.

    Parámetros:
    - palabra: Cadena de texto que representa la palabra.

    Retorna:
    - caracteristicas: Diccionario con características morfológicas.
    """
    caracteristicas = {
        'sufijo': palabra[-2:] if len(palabra) > 2 else palabra,
        'prefijo': palabra[:2] if len(palabra) > 2 else palabra,
        'mayuscula': palabra[0].isupper(),
        'numero': any(c.isdigit() for c in palabra),
        'puntuacion': not palabra.isalnum()
    }
    return caracteristicas

if __name__ == "__main__":
    ruta_corpus = "corpus_etiquetado.txt"
    corpus = cargar_corpus_desde_archivo(ruta_corpus)
    
    prob_emision, prob_transicion, etiquetas = recalcular_probabilidades_normalizadas(corpus)
    
    oracion = "habla con el enfermo grave de trasplantes .".lower().split()
    etiquetas_predichas, matriz_viterbi = viterbi(oracion, etiquetas, prob_emision, prob_transicion)
    
    guardar_resultados(prob_emision, prob_transicion, matriz_viterbi, oracion, etiquetas)
    analizar_resultados(oracion, etiquetas_predichas)
