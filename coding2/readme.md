# Semantic Graph Construction from Text

## 1. Problema

Dado un texto y un conjunto de términos relevantes, el objetivo es descubrir la estructura semántica identificando:

- El peso total de la estructura que conecta todos los términos
- El término más jerárquico
- Los cinco términos más centrales

El reto es ir más allá de la frecuencia y capturar relaciones semánticas entre conceptos.

## 2. Enfoque General

La solución sigue un enfoque basado en grafos:

1. Convertir términos en vectores numéricos (embedding)
2. Medir similitud semántica entre términos
3. Construir un grafo con esas similitudes
4. Extraer su estructura mínima representativa (MST)
5. Calcular jerarquía semántica sobre esa estructura

En resumen: texto, vectores, grafo, árbol, jerarquía.

## 3. Detección y Normalización

Normalizamos el texto (minúsculas, sin acentos ni puntuación) y detectamos términos relevantes como nombres propios y siglas. Cada término se convierte en un nodo del grafo.

## 4. Embedding

Cada término se representa mediante un embedding TF-IDF basado en trigramas de caracteres, normalizado a norma 1.

Elegimos este enfoque porque:

- Funciona con nombres propios y siglas que no están en vocabularios pre-entrenados
- No depende de modelos externos como Word2Vec o BERT
- Es determinista y reproducible

El embedding define la geometría del problema: determina qué tan cerca o lejos están dos términos.

## 5. Cálculo de Similitud

Calculamos la similitud coseno entre todos los pares de términos.

Complejidad: O(n² · d)

Este es el cuello de botella del pipeline, asumido en el enunciado. Para limitar el tamaño del grafo, conectamos cada término solo con sus k vecinos más cercanos.

## 6. Árbol de Expansión Mínima

Del grafo disperso obtenemos un MST que:

- Conecta todos los términos
- Minimiza el peso total
- Elimina conexiones redundantes

Este árbol representa la estructura semántica mínima del texto.

## 7. Jerarquía con Tree DP

La jerarquía semántica se define como:

```
S(u) = Σᵥ dist(u, v)
```

La suma de distancias desde un nodo u hacia todos los demás.

La solución naive ejecuta Dijkstra desde cada nodo con costo O(n²). Nuestra optimización aprovecha que el MST es un árbol y aplica Tree DP con dos recorridos:

1. Post-order: calcular tamaños de subárbol
2. Pre-order: propagar resultados con la fórmula:

```
S(hijo) = S(padre) + peso × (n - 2 × subtree_size)
```

Esto reduce el cálculo a O(n). Esta es la optimización principal.

## 8. Resultados

La solución produce:

- Peso total del MST
- Término más jerárquico (menor S(u))
- Top 5 de términos centrales

Estos resultados reflejan importancia estructural, no solo frecuencia.

## 9. Complejidad

| Etapa | Tiempo | Espacio |
|-------|--------|---------|
| Normalización | O(L) | O(L) |
| Detección | O(L·n) | O(n) |
| Embeddings | O(n·d) | O(n·d) |
| Similitud | O(n²·d) | O(n²) |
| MST | O(m·log n) | O(n) |
| Jerarquía | O(n) | O(n) |

Término dominante: O(n²·d) por el cálculo de similitudes.

## 10. Conclusión

- El embedding define la semántica del método
- El cálculo de similitudes es el cuello de botella esperado
- El MST revela la estructura conceptual mínima
- Tree DP permite calcular jerarquía en O(n)

La solución es exacta, eficiente y correcta.

