# Semantic Graph Construction from Text

## 1. Problema

Dado un texto y un conjunto de términos relevantes, el objetivo es descubrir la estructura semántica del texto, identificando:

- El peso total de la estructura que conecta todos los términos
- El término más jerárquico
- Los cinco términos más centrales

El reto principal es ir más allá de la frecuencia y capturar **relaciones semánticas** entre conceptos.

---

## 2. Enfoque General

La solución sigue un enfoque basado en grafos:

1. Convertimos términos en vectores numéricos (embedding)
2. Medimos similitud semántica entre términos
3. Construimos un grafo con esas similitudes
4. Extraemos su estructura mínima representativa (MST)
5. Calculamos jerarquía semántica sobre esa estructura

**En resumen:** `texto → vectores → grafo → árbol → jerarquía`

---

## 3. Detección y Normalización de Términos

Primero normalizamos el texto (minúsculas, sin acentos ni puntuación) y detectamos términos relevantes como nombres propios y siglas.

Cada término detectado se convierte en un **nodo del grafo**.

---

## 4. Embedding: la clave de la solución

Cada término se representa mediante un **embedding TF-IDF basado en trigramas de caracteres**, normalizado a norma 1.

Elegimos este enfoque porque:

- Funciona bien con nombres propios y siglas (ej: "Cristian Brokate")
- No depende de vocabularios pre-entrenados como Word2Vec o BERT
- Es determinista, reproducible y adecuado para textos especializados

👉 **El embedding define la geometría del problema:** determina qué tan "cerca" o "lejos" están dos términos y, por lo tanto, controla toda la estructura del grafo.

---

## 5. Cálculo de Similitud y Cuello de Botella

Usando los embeddings, calculamos la **similitud coseno** entre todos los pares de términos.

**Complejidad:** `O(n² · d)`

Este paso es el **principal cuello de botella** del pipeline y está explícitamente asumido en el enunciado.

Para limitar el tamaño del grafo, conectamos cada término solo con sus **k vecinos más cercanos exactos**, construyendo un grafo disperso sin perder exactitud.

---

## 6. Árbol de Expansión Mínima (MST)

Del grafo disperso obtenemos un **árbol de expansión mínima**, que:

- Conecta todos los términos
- Minimiza el peso total
- Elimina conexiones redundantes

Este árbol representa la **columna vertebral semántica** del texto.

---

## 7. Optimización Principal: Jerarquía con Tree DP

La jerarquía semántica se define como:

```
S(u) = Σᵥ dist(u, v)
```

Es decir, la suma de distancias desde un nodo `u` hacia todos los demás nodos `v`.

### Solución naive
Ejecutar Dijkstra desde cada nodo del MST → **O(n²)**

### Nuestra optimización
Aprovechando que el MST es un **árbol**, aplicamos **Tree Dynamic Programming** con dos recorridos:

1. **Post-order:** calcular tamaños de subárbol
2. **Pre-order:** propagar resultados usando la fórmula:

```
S(hijo) = S(padre) + peso × (n - 2 × subtree_size)
```

Esto reduce el cálculo de la jerarquía a **O(n)**, manteniendo exactitud total.

👉 **Esta es la optimización clave del trabajo.**

---

## 8. Resultados

La solución produce:

- El **peso total del MST**
- El **término más jerárquico** (menor S(u))
- El **top 5 de términos centrales**

Estos resultados reflejan **importancia estructural**, no solo frecuencia.

---

## 9. Análisis de Complejidad

| Etapa | Complejidad | Espacio |
|-------|-------------|---------|
| Normalización | O(L) | O(L) |
| Detección términos | O(L·n) | O(n) |
| Embeddings TF-IDF | O(n·d) | O(n·d) |
| Matriz similitud | **O(n²·d)** | O(n²) |
| MST | O(m·log n) | O(n) |
| Jerarquía Tree DP | **O(n)** | O(n) |

**Término dominante:** `O(n²·d)` por el cálculo de similitudes.

---

## 10. Conclusión

- El **embedding** es la base semántica del método
- El **cálculo de similitudes** es el cuello de botella esperado
- El **MST** revela la estructura conceptual mínima
- La **optimización con Tree DP** permite calcular jerarquía en O(n)

👉 **La solución es exacta, eficiente y defendible académicamente.**

---

### Frase final

> *"Transformamos texto en geometría y usamos grafos para extraer la estructura semántica y la jerarquía conceptual del contenido."*

