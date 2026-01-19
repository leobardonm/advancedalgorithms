import re
import math
import unicodedata
from collections import Counter, defaultdict

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl

# --------------------------------------------------
# 1. Normalización de texto
# --------------------------------------------------

def normalize(text: str) -> str:
    text = "".join(
        c for c in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(c)
    )
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# --------------------------------------------------
# 2. Extracción de términos (heurística simple)
# --------------------------------------------------

def extract_terms(text: str):
    # Secuencias con mayúsculas (nombres propios, siglas)
    candidates = re.findall(r"\b(?:[A-Z][a-z]+|[A-Z]{2,})(?:\s(?:[A-Z][a-z]+|[A-Z]{2,}))*\b", text)
    return candidates


# --------------------------------------------------
# 3. Conteo de frecuencia y posición
# --------------------------------------------------

def count_terms(text: str, terms):
    text_norm = normalize(text)
    stats = {}

    for t in terms:
        key = normalize(t)
        matches = list(re.finditer(rf"\b{re.escape(key)}\b", text_norm))
        if matches:
            stats[t] = {
                "freq": len(matches),
                "first_pos": matches[0].start()
            }
    return stats


# --------------------------------------------------
# 4. TF-IDF sobre trigramas de caracteres
# --------------------------------------------------

def char_ngrams(s, n=3):
    s = s.replace(" ", "_")
    return [s[i:i+n] for i in range(len(s)-n+1)] if len(s) >= n else []

def build_embeddings(terms):
    counts = [Counter(char_ngrams(normalize(t))) for t in terms]
    df = Counter()

    for c in counts:
        for g in c:
            df[g] += 1

    vocab = list(df.keys())
    idx = {g:i for i,g in enumerate(vocab)}
    N = len(terms)

    X = np.zeros((N, len(vocab)))

    for i, c in enumerate(counts):
        for g, tf in c.items():
            X[i, idx[g]] = tf * (math.log((N+1)/(df[g]+1)) + 1)

    # Normalización
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return X / norms


# --------------------------------------------------
# 5. Grafo semántico disperso (top-k vecinos)
# --------------------------------------------------

def build_graph(terms, vectors, k=5):
    sim = vectors @ vectors.T
    np.fill_diagonal(sim, -1)

    G = nx.Graph()
    G.add_nodes_from(terms)

    for i, u in enumerate(terms):
        neighbors = np.argsort(-sim[i])[:k]
        for j in neighbors:
            v = terms[j]
            w = 1 - sim[i, j]
            G.add_edge(u, v, weight=w)

    return G


# --------------------------------------------------
# 6. Árbol de expansión mínima
# --------------------------------------------------

def minimum_spanning_structure(G):
    return nx.minimum_spanning_tree(G, weight="weight")


# --------------------------------------------------
# 7. Jerarquía semántica
# --------------------------------------------------

def semantic_hierarchy(tree):
    scores = {}
    for u, dists in nx.all_pairs_dijkstra_path_length(tree, weight="weight"):
        scores[u] = sum(dists.values())
    best = min(scores, key=scores.get)
    return best, scores


# --------------------------------------------------
# 8. Visualización
# --------------------------------------------------

def visualize(G, T, highlight=None):
    pos = nx.spring_layout(G, seed=7)

    def draw(ax, graph, title):
        weights = [graph[u][v]["weight"] for u,v in graph.edges()]
        cmap = plt.cm.viridis
        norm = mpl.colors.Normalize(vmin=min(weights), vmax=max(weights))

        node_colors = ["red" if n == highlight else "blue" for n in graph.nodes()]

        nx.draw_networkx_nodes(graph, pos, node_color=node_colors, ax=ax)
        nx.draw_networkx_labels(graph, pos, font_size=9, ax=ax)
        nx.draw_networkx_edges(
            graph, pos,
            edge_color=[cmap(norm(w)) for w in weights],
            width=2,
            ax=ax
        )
        ax.set_title(title)
        ax.axis("off")

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    draw(axs[0], G, "Grafo Semántico Disperso")
    draw(axs[1], T, "Árbol de Expansión Mínima")
    plt.show()


# --------------------------------------------------
# 9. Pipeline completo
# --------------------------------------------------

def run_pipeline(text):
    raw_terms = extract_terms(text)
    stats = count_terms(text, raw_terms)

    terms = list(stats.keys())
    vectors = build_embeddings(terms)

    G = build_graph(terms, vectors, k=5)
    T = minimum_spanning_structure(G)

    hierarchical, scores = semantic_hierarchy(T)

    visualize(G, T, highlight=hierarchical)

    print("Término más jerárquico:", hierarchical)
    print("\nTop 5 términos centrales:")
    for t, s in sorted(scores.items(), key=lambda x: x[1])[:5]:
        print(f"{t:30s} S={s:.3f} freq={stats[t]['freq']}")
