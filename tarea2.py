def es_seguro(v, grafo, colores, c):
    for i in range(len(grafo)):
        if grafo[v][i] == 1 and colores[i] == c:
            return False
    return True

def colorear_grafo(grafo, k, colores, v=0):
    if v == len(grafo):
        return True

    for c in range(1, k + 1):
        if es_seguro(v, grafo, colores, c):
            colores[v] = c
            if colorear_grafo(grafo, k, colores, v + 1):
                return True
            colores[v] = 0

    return False

import time

grafos = [
    # Grafo pequeño (5 vértices)
    [
        [0,1,1,0,1],
        [1,0,1,1,0],
        [1,1,0,1,0],
        [0,1,1,0,1],
        [1,0,0,1,0]
    ],
    # Grafo mediano (8 vértices)
    [
        [0,1,1,0,0,0,0,1],
        [1,0,1,1,0,0,0,0],
        [1,1,0,1,1,0,0,0],
        [0,1,1,0,1,1,0,0],
        [0,0,1,1,0,1,1,0],
        [0,0,0,1,1,0,1,1],
        [0,0,0,0,1,1,0,1],
        [1,0,0,0,0,1,1,0]
    ],
    # Grafo grande (12 vértices)
    [[0 if i==j else 1 for j in range(12)] for i in range(12)]
]

k = 3

for i, grafo in enumerate(grafos):
    colores = [0] * len(grafo)
    inicio = time.time()
    resultado = colorear_grafo(grafo, k, colores)
    fin = time.time()
    print(f"Grafo {i+1}: vértices={len(grafo)}, resultado={resultado}, tiempo={fin-inicio:.5f}s")


import time
import matplotlib.pyplot as plt

def generar_grafo_completo(n):
    return [[0 if i == j else 1 for j in range(n)] for i in range(n)]

tamanos = [4, 6, 8, 10]
tiempos = []
k = 3

for n in tamanos:
    grafo = generar_grafo_completo(n)
    colores = [0] * n
    inicio = time.time()
    colorear_grafo(grafo, k, colores)
    fin = time.time()
    tiempos.append(fin - inicio)

plt.plot(tamanos, tiempos, marker='o')
plt.xlabel("Número de vértices")
plt.ylabel("Tiempo de ejecución (segundos)")
plt.title("Coloreo de grafos usando Backtracking")
plt.grid(True)
plt.show()
