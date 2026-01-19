import re
import math
import unicodedata
from collections import Counter, defaultdict, deque
from typing import List, Dict, Tuple

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl


# ============================================================
# 1. Normalización
# ============================================================

def normalize_text(s: str) -> str:
    s = "".join(c for c in unicodedata.normalize("NFKD", s)
                if not unicodedata.combining(c))
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s\-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_term(s: str) -> str:
    s = normalize_text(s)
    s = s.replace("-", " ")
    return re.sub(r"\s+", " ", s).strip()


# ============================================================
# 2. Extracción simple de términos (nombres propios / siglas)
# ============================================================

def extract_terms(text: str) -> List[str]:
    return re.findall(
        r"\b(?:[A-Z][a-z]+|[A-Z]{2,})(?:\s(?:[A-Z][a-z]+|[A-Z]{2,}))*\b",
        text
    )


# ============================================================
# 3. Conteo de frecuencia y primera aparición
# ============================================================

def count_terms(text: str, terms: List[str]) -> Dict[str, Dict]:
    text_norm = normalize_text(text)
    stats = {}

    for t in terms:
        key = normalize_term(t)
        matches = list(re.finditer(rf"\b{re.escape(key)}\b", text_norm))
        if matches:
            stats[t] = {
                "freq": len(matches),
                "first_pos": matches[0].start()
            }
    return stats


# ============================================================
# 4. Embeddings TF-IDF con trigramas de caracteres
# ============================================================

def char_ngrams(s: str, n: int = 3) -> List[str]:
    s = s.replace(" ", "_")
    if len(s) < n:
        return []
    return [s[i:i+n] for i in range(len(s) - n + 1)]


def build_embeddings(terms: List[str]) -> np.ndarray:
    counts = [Counter(char_ngrams(normalize_term(t))) for t in terms]
    df = Counter()

    for c in counts:
        for g in c:
            df[g] += 1

    vocab = list(df.keys())
    idx = {g: i for i, g in enumerate(vocab)}
    N = len(terms)

    X = np.zeros((N, len(vocab)))

    for i, c in enumerate(counts):
        for g, tf in c.items():
            idf = math.log((N + 1) / (df[g] + 1)) + 1
            X[i, idx[g]] = tf * idf

    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return X / norms


# ============================================================
# 5. Grafo k-NN exacto (similitud coseno)
# ============================================================

def build_sparse_graph(labels: List[str], V: np.ndarray, k: int = 5) -> nx.Graph:
    sim = V @ V.T
    np.fill_diagonal(sim, -1)

    G = nx.Graph()
    G.add_nodes_from(labels)

    for i, u in enumerate(labels):
        neighbors = np.argsort(-sim[i])[:k]
        for j in neighbors:
            v = labels[j]
            w = 1.0 - sim[i, j]
            if not G.has_edge(u, v):
                G.add_edge(u, v, weight=w, sim=sim[i, j])

    # Garantizar conectividad
    if not nx.is_connected(G):
        comps = list(nx.connected_components(G))
        for a in range(len(comps) - 1):
            u = next(iter(comps[a]))
            v = next(iter(comps[a + 1]))
            i, j = labels.index(u), labels.index(v)
            w = 1.0 - sim[i, j]
            G.add_edge(u, v, weight=w, sim=sim[i, j])

    return G


# ============================================================
# 6. Árbol de expansión mínima
# ============================================================

def compute_mst(G: nx.Graph) -> nx.Graph:
    return nx.minimum_spanning_tree(G, weight="weight")


# ============================================================
# 7. Jerarquía semántica con Tree DP (O(n))
# ============================================================

def semantic_hierarchy_tree_dp(tree: nx.Graph) -> Tuple[str, Dict[str, float]]:
    root = next(iter(tree.nodes))
    n = tree.number_of_nodes()

    adj = defaultdict(list)
    for u, v, d in tree.edges(data=True):
        w = d["weight"]
        adj[u].append((v, w))
        adj[v].append((u, w))

    subtree_size = {}
    subtree_dist = {}
    parent = {root: None}

    # Post-order DFS
    stack = [(root, False)]
    order = []

    while stack:
        node, visited = stack.pop()
        if visited:
            order.append(node)
        else:
            stack.append((node, True))
            for nei, _ in adj[node]:
                if nei != parent.get(node):
                    parent[nei] = node
                    stack.append((nei, False))

    for u in order:
        size = 1
        dist = 0.0
        for v, w in adj[u]:
            if parent.get(v) == u:
                size += subtree_size[v]
                dist += subtree_dist[v] + w * subtree_size[v]
        subtree_size[u] = size
        subtree_dist[u] = dist

    S = {root: subtree_dist[root]}
    queue = deque([root])

    while queue:
        u = queue.popleft()
        for v, w in adj[u]:
            if v not in S:
                S[v] = S[u] + w * (n - 2 * subtree_size[v])
                queue.append(v)

    best = min(S, key=S.get)
    return best, S


# ============================================================
# 8. Visualización
# ============================================================

def visualize(G: nx.Graph, MST: nx.Graph, highlight: str = None):
    """Visualiza el grafo disperso y el MST con heatmap de pesos."""
    pos = nx.spring_layout(G, seed=42)
    
    # Obtener pesos para normalización
    w_g = [G[u][v]["weight"] for u, v in G.edges()]
    w_mst = [MST[u][v]["weight"] for u, v in MST.edges()]
    
    if not w_g and not w_mst:
        print("No hay aristas para visualizar.")
        return
    
    w_min = min(min(w_g, default=0), min(w_mst, default=0))
    w_max = max(max(w_g, default=1), max(w_mst, default=1))
    
    if abs(w_max - w_min) < 1e-12:
        w_min, w_max = 0.0, 1.0
    
    cmap = plt.cm.viridis
    norm = mpl.colors.Normalize(vmin=w_min, vmax=w_max)
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    
    # Colores de nodos
    node_colors = [
        "tab:red" if (highlight and n == highlight) else "tab:blue"
        for n in G.nodes()
    ]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Grafo disperso
    ax1.set_title("Grafo Semántico Disperso (k-NN)")
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color=node_colors, ax=ax1)
    nx.draw_networkx_labels(G, pos, font_size=8, ax=ax1)
    edge_colors_g = [cmap(norm(G[u][v]["weight"])) for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors_g, width=2, alpha=0.8, ax=ax1)
    ax1.axis("off")
    fig.colorbar(sm, ax=ax1, fraction=0.046, pad=0.04, label="Peso = 1 − similitud")
    
    # MST
    ax2.set_title("Árbol de Expansión Mínima (MST)")
    nx.draw_networkx_nodes(MST, pos, node_size=500, node_color=node_colors, ax=ax2)
    nx.draw_networkx_labels(MST, pos, font_size=8, ax=ax2)
    edge_colors_mst = [cmap(norm(MST[u][v]["weight"])) for u, v in MST.edges()]
    nx.draw_networkx_edges(MST, pos, edge_color=edge_colors_mst, width=3, alpha=0.9, ax=ax2)
    ax2.axis("off")
    fig.colorbar(sm, ax=ax2, fraction=0.046, pad=0.04, label="Peso = 1 − similitud")
    
    plt.tight_layout()
    plt.show()


# ============================================================
# 9. Pipeline completo
# ============================================================

def run_pipeline(text: str, topk: int = 5, visualize_graph: bool = True):
    raw_terms = extract_terms(text)
    stats = count_terms(text, raw_terms)

    labels = list(stats.keys())
    V = build_embeddings(labels)

    G = build_sparse_graph(labels, V, k=topk)
    MST = compute_mst(G)

    hier, scores = semantic_hierarchy_tree_dp(MST)
    
    # Agregar frecuencia a los nodos
    for lab in labels:
        G.nodes[lab]["freq"] = stats[lab]["freq"]
        G.nodes[lab]["first_pos"] = stats[lab]["first_pos"]

    total_weight = sum(MST[u][v]["weight"] for u, v in MST.edges())

    # Visualización
    if visualize_graph:
        visualize(G, MST, highlight=hier)

    # Resultados
    print("\n" + "=" * 60)
    print("RESULTADOS")
    print("=" * 60)
    print(f"Términos detectados: {len(labels)}")
    print(f"Peso total MST: {total_weight:.6f}")
    print(f"Término más jerárquico: {hier}")
    print("\nTop 5 términos centrales:")
    for t, s in sorted(scores.items(), key=lambda x: x[1])[:5]:
        print(f"  {t:30s}  S={s:.4f}  freq={stats[t]['freq']}")

    return G, MST, hier, scores, total_weight


# ============================================================
# 10. Ejemplo de uso
# ============================================================

if __name__ == "__main__":
    TEXT = """
    Unmasking information manipulation: A quantitative approach to detecting 
    copy-pasta, rewording, and translation on social media.

    Manon Richard, Lisa Giordani, Cristian Brokate, Jean Lienard.

    Abstract. This study proposes a methodology for identifying three techniques 
    used in foreign-operated information manipulation campaigns: copy-pasta, 
    rewording, and translation. The approach, called the "3 Delta-space duplicate 
    methodology", quantifies three aspects of messages: semantic meaning, 
    grapheme-level wording, and language.

    The method is validated using a synthetic dataset generated with ChatGPT and 
    DeepL, and then applied to a Twitter Transparency dataset (2021) about 
    Venezuelan actors. The method identifies all three types of inauthentic 
    duplicates in the synthetic dataset and uncovers duplicates in the Twitter 
    dataset across political, commercial, and entertainment contexts.

    The analysis identifies political boosting, commercial messages for alcoholic 
    beverages, and entertainment messages related to "The Walking Dead". 
    Coordinated inauthentic behavior (CIB) refers to synchronized efforts to 
    shape online discourse through repeated activity using AI tools.
    """
    
    run_pipeline(TEXT, topk=5, visualize_graph=True)
