import time
import itertools
import matplotlib.pyplot as plt

def calcular_lmax(orden, trabajos):
    tiempo = 0
    lmax = float('-inf')
    for i in orden:
        tiempo += trabajos[i][0]  # p_i
        lmax = max(lmax, tiempo - trabajos[i][1])  # f_i - d_i
    return lmax

def branch_and_bound(trabajos):
    n = len(trabajos)
    mejor_lmax = float('inf')
    mejor_orden = None

    for perm in itertools.permutations(range(n)):
        lmax = calcular_lmax(perm, trabajos)
        if lmax < mejor_lmax:
            mejor_lmax = lmax
            mejor_orden = perm

    return mejor_lmax, mejor_orden

conjuntos = [
    # n = 4
    [(3, 6), (2, 8), (4, 9), (1, 5)],
    # n = 5
    [(2, 6), (1, 4), (3, 8), (2, 7), (4, 10)],
    # n = 6
    [(3, 7), (2, 5), (4, 10), (1, 6), (2, 8), (3, 9)],
    # n = 7
    [(2, 4), (3, 7), (1, 5), (4, 9), (2, 6), (3, 8), (1, 10)]
]

for i, trabajos in enumerate(conjuntos):
    inicio = time.time()
    lmax, orden = branch_and_bound(trabajos)
    fin = time.time()
    print(f"Conjunto {i+1}: n={len(trabajos)}, Lmax={lmax}, tiempo={fin-inicio:.5f}s")


tamanos = [4, 5, 6, 7]
tiempos = []

for trabajos in conjuntos:
    inicio = time.time()
    branch_and_bound(trabajos)
    fin = time.time()
    tiempos.append(fin - inicio)

plt.plot(tamanos, tiempos, marker='o')
plt.xlabel("Número de trabajos")
plt.ylabel("Tiempo de ejecución (segundos)")
plt.title("Branch and Bound para minimizar Lmax")
plt.grid(True)
plt.show()