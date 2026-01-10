from typing import List, Tuple, Callable
import time
import statistics
import csv
import os
import matplotlib.pyplot as plt


# ============================================================
# 1. METODO GREEDY (LEOBARDO)
# ============================================================

def greedy_knapsack(
    products: List[Tuple[str, str, int, int]],
    capacity_W: int
) -> Tuple[int, List[str]]:
    """
    Heurística greedy para knapsack 0/1.
    Ordena por valor/peso, luego por menor peso y por id.
    """

    if capacity_W <= 0:
        return 0, []

    products_sorted = sorted(
        products,
        key=lambda p: (-p[3] / p[2], p[2], p[0])
    )

    remaining_capacity = capacity_W
    total_value = 0
    chosen_ids = []

    for product_id, _, weight, value in products_sorted:
        if weight <= remaining_capacity:
            chosen_ids.append(product_id)
            remaining_capacity -= weight
            total_value += value

    return total_value, chosen_ids


# ============================================================
# 2. METODO PROGRAMACION DINAMICA (DIEGO)
# ============================================================

def dp_knapsack(
    products: List[Tuple[str, str, int, int]],
    capacity_W: int
) -> Tuple[int, List[str]]:
    """
    Knapsack 0/1 usando programación dinámica. 
    Procesa cada item una sola vez y recorre capacidades en orden descendente.
    """

    if capacity_W <= 0:
        return 0, []

    n = len(products)

    # dp[w] = mejor valor alcanzable con capacidad w
    dp = [0] * (capacity_W + 1)

    # choice[i][w] = True si el producto i se toma al construir dp[w]
    choice = [[False] * (capacity_W + 1) for _ in range(n)]

    for i in range(n):
        product_id, name, weight, value = products[i]

        for w in range(capacity_W, weight - 1, -1):
            if dp[w - weight] + value > dp[w]:
                dp[w] = dp[w - weight] + value
                choice[i][w] = True

    # reconstrucción de la solución
    chosen_ids = []
    w = capacity_W

    for i in range(n - 1, -1, -1):
        if choice[i][w]:
            product_id, name, weight, value = products[i]
            chosen_ids.append(product_id)
            w -= weight

    chosen_ids.reverse()
    return dp[capacity_W], chosen_ids


# ============================================================
# 3. METODO BACKTRACKING + BRANCH AND BOUND (SANTIAGO)
# ============================================================

def backtracking_knapsack(
    products: List[Tuple[str, str, int, int]],
    capacity_W: int
) -> Tuple[int, List[str]]:
    """
    Solver exacto usando backtracking con branch-and-bound para Knapsack 0/1.
    
    Algoritmo:
    - Explora items en orden no creciente de densidad (valor/peso)
    - Mantiene la mejor solución factible encontrada
    - Poda ramas usando cota superior fraccional (admisible)
    
    Args:
        products: Lista de tuplas (id, nombre, peso, valor)
        capacity_W: Capacidad máxima en kilogramos
    
    Returns:
        Tupla (mejor_valor, lista_ids_elegidos)
    """
    if capacity_W <= 0 or not products:
        return 0, []
    
    # Ordenar por densidad decreciente para mejor poda
    items = sorted(
        products,
        key=lambda x: x[3] / x[2],  # valor / peso
        reverse=True
    )
    
    n = len(items)
    best_value = 0
    best_choice = []
    
    def upper_bound(start, current_weight, current_value):
        """
        Cota superior usando knapsack fraccional.
        Es una cota admisible (optimista) que permite podar ramas
        que no pueden mejorar la mejor solución actual.
        """
        bound = current_value
        remaining_capacity = capacity_W - current_weight
        
        for i in range(start, n):
            _, _, w, v = items[i]
            if w <= remaining_capacity:
                remaining_capacity -= w
                bound += v
            else:
                # Tomar fracción del item
                bound += v * (remaining_capacity / w)
                break
        return bound
    
    def backtrack(i, current_weight, current_value, chosen_ids):
        """
        Backtracking recursivo con branch-and-bound.
        """
        nonlocal best_value, best_choice
        
        # Actualizar mejor solución si la actual es mejor
        if current_value > best_value:
            best_value = current_value
            best_choice = chosen_ids.copy()
        
        # Caso base: no hay más items
        if i == n:
            return
        
        # Poda: si la cota superior no puede mejorar, cortar
        if upper_bound(i, current_weight, current_value) <= best_value:
            return
        
        pid, _, w, v = items[i]
        
        # Rama 1: Tomar el item i (si cabe)
        if current_weight + w <= capacity_W:
            chosen_ids.append(pid)
            backtrack(i + 1, current_weight + w, current_value + v, chosen_ids)
            chosen_ids.pop()
        
        # Rama 2: No tomar el item i
        backtrack(i + 1, current_weight, current_value, chosen_ids)
    
    # Iniciar backtracking desde el item 0
    backtrack(0, 0, 0, [])
    return best_value, best_choice


# ============================================================
# 4. CATALOGOS Y CAPACIDADES
# ============================================================

C1 = [
    ("P01","Chips Box",1,6),("P02","Soda Crate",2,11),("P03","Candy Bulk",3,16),
    ("P04","Water Pack",4,21),("P05","Fruit Crate",5,26),("P06","Ice Cream Bin",6,31),
    ("P07","BBQ Sauce Case",7,36),("P08","Snack Variety",8,40),
    ("P09","Cleaning Supplies",9,45),("P10","First-Aid Kits",10,50)
]

C2 = [
    ("Q10","Assorted Gadgets",10,60),("Q20","Party Drinks Pallet",20,100),
    ("Q30","Outdoor Grill",30,120),("Q35","Mini Freezer",35,130),
    ("Q40","Tool Chest",40,135),("Q45","Camp Bundle",45,140),
    ("Q50","Generator",50,150)
]

CAPACITIES = [0, 20, 35, 50, 65, 80, 95, 110, 140]

CATALOGS = {
    "C1": C1,
    "C2": C2
}


# ============================================================
# 5. BENCHMARKING
# ============================================================

N_RUNS = 5
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

METHODS: dict[str, Callable] = {
    "Greedy": greedy_knapsack,
    "DP": dp_knapsack,
    "Backtracking": backtracking_knapsack
}


def benchmark_method(method, products, capacity):
    times = []
    value = None

    for _ in range(N_RUNS):
        start = time.perf_counter()
        value, chosen = method(products, capacity)
        end = time.perf_counter()
        times.append((end - start) * 1000)

    return statistics.median(times), value


def run_all_experiments():
    raw_rows = []

    for catalog_name, catalog in CATALOGS.items():
        for capacity in CAPACITIES:
            for method_name, method in METHODS.items():
                for run in range(N_RUNS):
                    start = time.perf_counter()
                    value, chosen = method(catalog, capacity)
                    end = time.perf_counter()

                    raw_rows.append([
                        catalog_name,
                        capacity,
                        method_name,
                        run,
                        (end - start) * 1000,
                        value
                    ])

    # guardar CSV con tiempos crudos
    csv_path = os.path.join(RESULTS_DIR, "results_timings.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "catalog", "capacity", "method",
            "run", "runtime_ms", "value"
        ])
        writer.writerows(raw_rows)

    print(f"[OK] Resultados guardados en {csv_path}")


# ============================================================
# 6. GRAFICAS
# ============================================================

def generate_plots():
    for catalog_name, catalog in CATALOGS.items():

        runtimes = {m: [] for m in METHODS}
        values = {m: [] for m in METHODS}

        for capacity in CAPACITIES:
            for method_name, method in METHODS.items():
                median_time, value = benchmark_method(method, catalog, capacity)
                runtimes[method_name].append(median_time)
                values[method_name].append(value)

        # --- Runtime vs Capacity ---
        plt.figure()
        for method_name, times in runtimes.items():
            plt.plot(CAPACITIES, times, label=method_name)

        plt.yscale("log")
        plt.xlabel("Capacity")
        plt.ylabel("Median Runtime (ms)")
        plt.title(f"Runtime vs Capacity ({catalog_name})")
        plt.legend()
        plt.tight_layout()

        plt.savefig(os.path.join(RESULTS_DIR, f"runtime_{catalog_name}.png"), dpi=300)
        plt.savefig(os.path.join(RESULTS_DIR, f"runtime_{catalog_name}.pdf"))
        plt.close()

        # --- Greedy Value Gap ---
        if "Greedy" in values:
            greedy_vals = values["Greedy"]
            best_vals = [
                max(values[m][i] for m in values)
                for i in range(len(CAPACITIES))
            ]

            gaps = [b - g for b, g in zip(best_vals, greedy_vals)]

            plt.figure()
            plt.plot(CAPACITIES, gaps, marker="o")
            plt.axhline(0)
            plt.xlabel("Capacity")
            plt.ylabel("Best Value − Greedy Value")
            plt.title(f"Greedy Value Gap vs Capacity ({catalog_name})")
            plt.tight_layout()

            plt.savefig(os.path.join(RESULTS_DIR, f"gap_{catalog_name}.png"), dpi=300)
            plt.savefig(os.path.join(RESULTS_DIR, f"gap_{catalog_name}.pdf"))
            plt.close()

    print("[OK] Figuras generadas")


# ============================================================
# 7. MAIN
# ============================================================

if __name__ == "__main__":
    print("Running all knapsack experiments...")
    run_all_experiments()
    generate_plots()
    print("Done.")