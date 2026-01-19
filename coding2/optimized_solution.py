"""
Optimized Semantic Graph Construction from Text
================================================
Key optimizations over the baseline:
1. Approximate Nearest Neighbors (ANN) using ball trees - reduces O(n²·d) to O(n·log(n)·d)
2. Tree DP for hierarchy computation - reduces O(n²) to O(n)
3. Sparse matrix operations for memory efficiency
4. Vectorized batch processing for embeddings
5. Union-Find with path compression for connectivity checks

Author: Optimized Implementation
"""

import re
import math
import unicodedata
import time
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
import heapq

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl

# Memory profiling
try:
    import tracemalloc
except Exception:
    tracemalloc = None

try:
    import resource
except Exception:
    resource = None


# =============================================================================
# PROFILING UTILITIES
# =============================================================================
def _bytes_to_mb(x: int) -> float:
    return float(x) / (1024.0 * 1024.0)


def _rss_mb() -> Optional[float]:
    if resource is None:
        return None
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if ru > 10_000_000:
        return _bytes_to_mb(ru)
    return ru / 1024.0


@dataclass
class ProfileRecord:
    stage: str
    time_s: float
    py_peak_mb: Optional[float] = None
    py_net_mb: Optional[float] = None
    rss_mb: Optional[float] = None
    rss_delta_mb: Optional[float] = None
    extras: Optional[Dict] = None


class Profiler:
    """Lightweight profiler for benchmarking."""
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled and tracemalloc is not None
        self.records: List[ProfileRecord] = []
        self._t0 = None
        self._snap0 = None
        self._rss0 = None
        self._name = None
        
    def start(self):
        if self.enabled:
            tracemalloc.start()
        self._rss0 = _rss_mb()
        
    def stop(self):
        if self.enabled:
            tracemalloc.stop()
            
    def begin(self, name: str):
        self._name = name
        self._t0 = time.perf_counter()
        self._rss0_stage = _rss_mb()
        if self.enabled:
            self._snap0 = tracemalloc.take_snapshot()
            
    def end(self, extras: Optional[Dict] = None):
        t1 = time.perf_counter()
        dt = t1 - (self._t0 or t1)
        
        rss1 = _rss_mb()
        rss_delta = None if (rss1 is None or self._rss0_stage is None) else (rss1 - self._rss0_stage)
        
        py_peak = None
        py_net = None
        if self.enabled:
            snap1 = tracemalloc.take_snapshot()
            stats = snap1.compare_to(self._snap0, "lineno")
            py_alloc = sum(s.size_diff for s in stats)
            _, peak = tracemalloc.get_traced_memory()
            py_peak = _bytes_to_mb(int(peak))
            py_net = _bytes_to_mb(int(py_alloc))
            
        self.records.append(ProfileRecord(
            stage=self._name,
            time_s=dt,
            py_peak_mb=py_peak,
            py_net_mb=py_net,
            rss_mb=rss1,
            rss_delta_mb=rss_delta,
            extras=extras
        ))
        
    def summary(self) -> str:
        lines = ["\n=== Profiling Summary ==="]
        total_time = sum(r.time_s for r in self.records)
        
        for r in self.records:
            parts = [f"{r.stage}: {r.time_s:.4f}s ({100*r.time_s/total_time:.1f}%)"]
            if r.py_peak_mb is not None:
                parts.append(f"peak={r.py_peak_mb:.2f}MB")
            if r.rss_mb is not None:
                parts.append(f"rss={r.rss_mb:.2f}MB")
            if r.extras:
                for k, v in r.extras.items():
                    parts.append(f"{k}={v}")
            lines.append("  " + " | ".join(parts))
            
        lines.append(f"\n  TOTAL TIME: {total_time:.4f}s")
        return "\n".join(lines)


# =============================================================================
# STAGE 1: TEXT NORMALIZATION (O(L + Σ|tᵢ|) time, O(L + n) space)
# =============================================================================
def strip_accents(s: str) -> str:
    """Remove accents from string."""
    return "".join(
        ch for ch in unicodedata.normalize("NFKD", s) 
        if not unicodedata.combining(ch)
    )


def normalize_text(s: str) -> str:
    """Normalize text: lowercase, remove accents/punctuation, collapse spaces."""
    s = strip_accents(s).lower()
    s = re.sub(r"[^a-z0-9\s\-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_term(s: str) -> str:
    """Normalize a term for matching."""
    s = normalize_text(s)
    s = s.replace("-", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


# =============================================================================
# STAGE 1: TERM EXTRACTION WITH AHO-CORASICK (O(L + n + matches))
# =============================================================================
class AhoCorasick:
    """
    Aho-Corasick automaton for O(L + n + matches) multi-pattern matching.
    Much faster than O(L * n) naive approach for many patterns.
    """
    
    def __init__(self):
        self.goto = [{}]  # goto function
        self.fail = [0]   # failure function
        self.output = [[]]  # output function (list of pattern indices)
        self.patterns = []
        
    def add_pattern(self, pattern: str, idx: int):
        """Add a pattern with associated index."""
        state = 0
        for ch in pattern:
            if ch not in self.goto[state]:
                self.goto[state][ch] = len(self.goto)
                self.goto.append({})
                self.fail.append(0)
                self.output.append([])
            state = self.goto[state][ch]
        self.output[state].append(idx)
        self.patterns.append(pattern)
        
    def build(self):
        """Build failure links using BFS."""
        from collections import deque
        queue = deque()
        
        # Initialize depth-1 states
        for ch, s in self.goto[0].items():
            queue.append(s)
            self.fail[s] = 0
            
        # BFS to build failure links
        while queue:
            r = queue.popleft()
            for ch, s in self.goto[r].items():
                queue.append(s)
                state = self.fail[r]
                while state and ch not in self.goto[state]:
                    state = self.fail[state]
                self.fail[s] = self.goto[state].get(ch, 0)
                if self.fail[s] != s:
                    self.output[s] = self.output[s] + self.output[self.fail[s]]
                    
    def search(self, text: str) -> List[Tuple[int, int, int]]:
        """
        Search text for all patterns.
        Returns: List of (start_pos, end_pos, pattern_idx)
        """
        results = []
        state = 0
        
        for i, ch in enumerate(text):
            while state and ch not in self.goto[state]:
                state = self.fail[state]
            state = self.goto[state].get(ch, 0)
            
            for pat_idx in self.output[state]:
                pat_len = len(self.patterns[pat_idx])
                start = i - pat_len + 1
                results.append((start, i + 1, pat_idx))
                
        return results


def extract_terms_pattern(text: str) -> List[str]:
    """Extract capitalized phrases as candidate terms."""
    tokens = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]+(?:[-'][A-Za-zÀ-ÖØ-öø-ÿ0-9]+)?", text)
    connectors = {
        "of", "the", "and", "de", "del", "la", "las", "los", 
        "da", "do", "dos", "das", "di", "van", "von", "y", "&"
    }
    
    def is_caps(tok: str) -> bool:
        if len(tok) >= 2 and tok.isupper():
            return True
        return tok[:1].isupper() and tok[1:].islower()
    
    phrases = []
    i = 0
    while i < len(tokens):
        if is_caps(tokens[i]):
            j = i + 1
            phrase = [tokens[i]]
            while j < len(tokens):
                t2 = tokens[j]
                if t2.lower() in connectors:
                    phrase.append(t2)
                    j += 1
                    continue
                if is_caps(t2):
                    phrase.append(t2)
                    j += 1
                    continue
                break
            while phrase and phrase[-1].lower() in connectors:
                phrase.pop()
            phrases.append(" ".join(phrase))
            i = j
        else:
            i += 1
            
    # Also capture acronyms
    phrases.extend(re.findall(r"\b[A-Z]{2,}\b", text))
    return phrases


def count_terms_optimized(text: str, term_keys: List[str]) -> Dict[str, Dict]:
    """
    Count term occurrences using Aho-Corasick for O(L + n + matches) complexity.
    """
    text_norm = normalize_text(text)
    
    # Build Aho-Corasick automaton
    ac = AhoCorasick()
    for idx, key in enumerate(term_keys):
        if key:
            ac.add_pattern(key, idx)
    ac.build()
    
    # Search all patterns at once
    matches = ac.search(text_norm)
    
    # Aggregate results
    term_matches = defaultdict(list)
    for start, end, pat_idx in matches:
        # Check word boundaries
        if start > 0 and text_norm[start-1].isalnum():
            continue
        if end < len(text_norm) and text_norm[end].isalnum():
            continue
        term_matches[term_keys[pat_idx]].append(start)
    
    # Build output
    out = {}
    for key, positions in term_matches.items():
        if positions:
            out[key] = {
                "freq": len(positions),
                "first_pos": min(positions),
            }
    return out


# =============================================================================
# STAGE 2: VECTORIZED EMBEDDINGS (O(n·d) time, O(n·d) space)
# =============================================================================
def char_ngrams_vectorized(terms: List[str], n: int = 3) -> Tuple[np.ndarray, int]:
    """
    Build character n-gram TF-IDF vectors in a vectorized manner.
    Returns normalized vectors and embedding dimension.
    """
    N = len(terms)
    
    # Process terms to get n-grams
    processed = [t.replace(" ", "_") for t in terms]
    
    # Collect all n-grams and compute document frequencies
    all_ngrams = set()
    term_ngrams = []
    
    for t in processed:
        if len(t) < n:
            ngrams = Counter([t] if t else [])
        else:
            ngrams = Counter(t[i:i+n] for i in range(len(t) - n + 1))
        term_ngrams.append(ngrams)
        all_ngrams.update(ngrams.keys())
    
    # Create feature index mapping
    features = sorted(all_ngrams)
    feat_idx = {g: i for i, g in enumerate(features)}
    d = len(features)
    
    if d == 0:
        return np.zeros((N, 1)), 1
    
    # Compute document frequencies
    df = np.zeros(d)
    for ngrams in term_ngrams:
        for g in ngrams:
            df[feat_idx[g]] += 1
    
    # Compute IDF (smoothed)
    idf = np.log((N + 1) / (df + 1)) + 1.0
    
    # Build TF-IDF matrix (sparse-like construction)
    V = np.zeros((N, d), dtype=np.float32)
    for i, ngrams in enumerate(term_ngrams):
        for g, tf in ngrams.items():
            V[i, feat_idx[g]] = tf * idf[feat_idx[g]]
    
    # Normalize to unit vectors
    norms = np.linalg.norm(V, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    V = V / norms
    
    return V, d


# =============================================================================
# STAGE 3: SPARSE GRAPH WITH APPROXIMATE NEAREST NEIGHBORS
# O(n·k·log(n)·d) instead of O(n²·d) using ball tree
# =============================================================================
class BallTree:
    """
    Simple Ball Tree for approximate nearest neighbor search.
    Reduces O(n²) similarity computation to O(n·log(n)) average case.
    """
    
    def __init__(self, data: np.ndarray, leaf_size: int = 40):
        self.data = data
        self.leaf_size = leaf_size
        self.n, self.d = data.shape
        self.tree = self._build(np.arange(self.n))
        
    def _build(self, indices: np.ndarray) -> dict:
        """Recursively build the ball tree."""
        if len(indices) <= self.leaf_size:
            return {"indices": indices, "leaf": True}
        
        points = self.data[indices]
        center = points.mean(axis=0)
        radius = np.max(np.linalg.norm(points - center, axis=1))
        
        # Split along dimension with maximum variance
        variances = np.var(points, axis=0)
        split_dim = np.argmax(variances)
        median = np.median(points[:, split_dim])
        
        left_mask = points[:, split_dim] <= median
        # Ensure we don't create empty children
        if left_mask.sum() == 0 or left_mask.sum() == len(indices):
            return {"indices": indices, "leaf": True}
        
        left_indices = indices[left_mask]
        right_indices = indices[~left_mask]
        
        return {
            "center": center,
            "radius": radius,
            "left": self._build(left_indices),
            "right": self._build(right_indices),
            "leaf": False
        }
    
    def query_neighbors(self, query_idx: int, k: int) -> List[Tuple[float, int]]:
        """Find k nearest neighbors for a query point (excluding itself)."""
        query = self.data[query_idx]
        heap = []  # max-heap (negative distances)
        
        def search(node):
            if node["leaf"]:
                for idx in node["indices"]:
                    if idx == query_idx:
                        continue
                    # Cosine similarity = dot product (unit vectors)
                    sim = float(np.dot(query, self.data[idx]))
                    dist = 1.0 - sim
                    
                    if len(heap) < k:
                        heapq.heappush(heap, (-dist, idx))
                    elif dist < -heap[0][0]:
                        heapq.heapreplace(heap, (-dist, idx))
                return
            
            # Check if we can prune this branch
            if len(heap) >= k:
                max_dist = -heap[0][0]
                # Ball pruning: if closest possible point is farther than max_dist
                center_dist = np.linalg.norm(query - node["center"])
                if center_dist - node["radius"] > max_dist:
                    return
            
            # Visit children
            search(node["left"])
            search(node["right"])
        
        search(self.tree)
        return [(-d, i) for d, i in sorted(heap, reverse=True)]


class UnionFind:
    """Union-Find with path compression and union by rank for O(α(n)) operations."""
    
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.components = n
        
    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]
    
    def union(self, x: int, y: int) -> bool:
        px, py = self.find(x), self.find(y)
        if px == py:
            return False
        # Union by rank
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        self.components -= 1
        return True
    
    def connected(self, x: int, y: int) -> bool:
        return self.find(x) == self.find(y)


def build_sparse_graph_optimized(
    labels: List[str], 
    V: np.ndarray, 
    k: int = 5
) -> Tuple[nx.Graph, float]:
    """
    Build sparse similarity graph using approximate nearest neighbors.
    
    Complexity: O(n·k·log(n)·d) average case vs O(n²·d) naive
    Space: O(n·k) for graph edges
    """
    n = len(labels)
    G = nx.Graph()
    G.add_nodes_from(labels)
    
    if n <= 1:
        return G, 0.0
    
    kk = min(k, n - 1)
    label_idx = {lab: i for i, lab in enumerate(labels)}
    
    # Build ball tree for ANN queries
    tree = BallTree(V, leaf_size=max(10, n // 20))
    
    # Find k-nearest neighbors for each point
    edges = set()
    for i in range(n):
        neighbors = tree.query_neighbors(i, kk)
        for dist, j in neighbors:
            if i < j:
                edges.add((i, j, dist))
            else:
                edges.add((j, i, dist))
    
    # Add edges to graph
    for i, j, dist in edges:
        sim = 1.0 - dist
        G.add_edge(labels[i], labels[j], weight=dist, sim=sim)
    
    # Ensure connectivity using Union-Find
    uf = UnionFind(n)
    sorted_edges = sorted(edges, key=lambda x: x[2])
    
    for i, j, _ in sorted_edges:
        uf.union(i, j)
    
    # If not connected, add minimum weight edges between components
    if uf.components > 1:
        # Compute full similarity only between disconnected components
        component_map = defaultdict(list)
        for i in range(n):
            component_map[uf.find(i)].append(i)
        
        components = list(component_map.values())
        
        while len(components) > 1:
            best_edge = None
            best_weight = float('inf')
            
            for ci in range(len(components)):
                for cj in range(ci + 1, len(components)):
                    # Find best edge between components
                    for i in components[ci]:
                        for j in components[cj]:
                            sim = float(np.dot(V[i], V[j]))
                            weight = 1.0 - sim
                            if weight < best_weight:
                                best_weight = weight
                                best_edge = (i, j, weight, sim)
            
            if best_edge:
                i, j, w, s = best_edge
                G.add_edge(labels[i], labels[j], weight=w, sim=s)
                # Merge components
                merged = components[0] + components[1]
                components = [merged] + components[2:]
    
    return G, 0.0


# =============================================================================
# STAGE 4: KRUSKAL'S MST WITH UNION-FIND (O(m·log(n)))
# =============================================================================
def minimum_spanning_tree_kruskal(G: nx.Graph) -> nx.Graph:
    """
    Compute MST using Kruskal's algorithm with Union-Find.
    Complexity: O(m·log(m)) = O(m·log(n)) for sorting, O(m·α(n)) for union-find
    """
    nodes = list(G.nodes())
    n = len(nodes)
    node_idx = {node: i for i, node in enumerate(nodes)}
    
    # Get all edges sorted by weight
    edges = [(G[u][v]["weight"], u, v) for u, v in G.edges()]
    edges.sort()
    
    uf = UnionFind(n)
    mst = nx.Graph()
    mst.add_nodes_from(nodes)
    
    # Copy node attributes
    for node in nodes:
        mst.nodes[node].update(G.nodes[node])
    
    for weight, u, v in edges:
        if uf.union(node_idx[u], node_idx[v]):
            mst.add_edge(u, v, **G[u][v])
            if mst.number_of_edges() == n - 1:
                break
    
    return mst


# =============================================================================
# STAGE 5: TREE DP FOR HIERARCHY (O(n) instead of O(n²))
# =============================================================================
def compute_hierarchy_tree_dp(tree: nx.Graph) -> Tuple[str, Dict[str, float], float]:
    """
    Compute S(u) = Σᵥ dist(u,v) for all nodes using tree DP.
    
    Key insight: For a tree, we can compute all S(u) in O(n) using two DFS passes:
    1. First pass (post-order): Compute subtree sizes and sum of distances within subtree
    2. Second pass (pre-order): Use parent's result to compute full S(u)
    
    Complexity: O(n) time, O(n) space
    """
    if tree.number_of_nodes() == 0:
        return "", {}, 0.0
    
    if tree.number_of_nodes() == 1:
        node = list(tree.nodes())[0]
        return node, {node: 0.0}, 0.0
    
    nodes = list(tree.nodes())
    root = nodes[0]
    
    # Build adjacency list with weights
    adj = defaultdict(list)
    for u, v, data in tree.edges(data=True):
        w = data.get("weight", 1.0)
        adj[u].append((v, w))
        adj[v].append((u, w))
    
    # First DFS: compute subtree_size and subtree_dist_sum
    subtree_size = {}
    subtree_dist_sum = {}  # sum of distances from node to all nodes in its subtree
    parent = {root: None}
    parent_weight = {root: 0.0}
    
    # Post-order traversal using iterative DFS
    visited = set()
    stack = [(root, False)]
    order = []
    
    while stack:
        node, processed = stack.pop()
        if processed:
            order.append(node)
            continue
        if node in visited:
            continue
        visited.add(node)
        stack.append((node, True))
        for neighbor, w in adj[node]:
            if neighbor not in visited:
                parent[neighbor] = node
                parent_weight[neighbor] = w
                stack.append((neighbor, False))
    
    # Process in post-order (leaves to root)
    for node in order:
        size = 1
        dist_sum = 0.0
        for neighbor, w in adj[node]:
            if parent.get(neighbor) == node:  # neighbor is child
                child_size = subtree_size[neighbor]
                child_dist_sum = subtree_dist_sum[neighbor]
                size += child_size
                # Distance to each node in child subtree = w + dist from child
                dist_sum += child_dist_sum + w * child_size
        subtree_size[node] = size
        subtree_dist_sum[node] = dist_sum
    
    n = len(nodes)
    
    # Second DFS: compute S(u) for all nodes
    S = {}
    
    # BFS from root
    from collections import deque
    queue = deque([root])
    S[root] = subtree_dist_sum[root]
    
    while queue:
        node = queue.popleft()
        for neighbor, w in adj[node]:
            if neighbor not in S:
                # neighbor is a child of node
                # S[neighbor] = S[node] 
                #   - (distance contribution from neighbor's subtree going through node)
                #   + (distance contribution from rest of tree going through node)
                #
                # Removing neighbor subtree from node: -subtree_dist_sum[neighbor] - w*subtree_size[neighbor]
                # Adding path from neighbor to rest: +w*(n - subtree_size[neighbor])
                # Then adding back neighbor subtree internal: subtree_dist_sum[neighbor]
                
                child_size = subtree_size[neighbor]
                S[neighbor] = (S[node] 
                              - subtree_dist_sum[neighbor] - w * child_size  # remove contribution through edge
                              + w * (n - child_size)  # add reverse contribution
                              + subtree_dist_sum[neighbor])  # internal distances unchanged
                
                # Simplify: S[neighbor] = S[node] + w * (n - 2 * child_size)
                S[neighbor] = S[node] + w * (n - 2 * child_size)
                queue.append(neighbor)
    
    # Find minimum
    best_node = min(S, key=S.get)
    total_weight = sum(tree[u][v]["weight"] for u, v in tree.edges())
    
    return best_node, S, total_weight


# =============================================================================
# VISUALIZATION
# =============================================================================
def visualize_graphs(G: nx.Graph, S: nx.Graph, highlight: Optional[str] = None):
    """Draw both graphs with heatmap coloring for edge weights."""
    pos = nx.spring_layout(G, seed=42)
    
    def get_weights(H):
        return np.array([H[u][v]["weight"] for u, v in H.edges()]) if H.number_of_edges() else np.array([])
    
    w_g = get_weights(G)
    w_s = get_weights(S)
    
    if w_g.size == 0 and w_s.size == 0:
        print("No edges to visualize.")
        return
    
    w_min = min(w_g.min() if w_g.size else float('inf'), w_s.min() if w_s.size else float('inf'))
    w_max = max(w_g.max() if w_g.size else 0, w_s.max() if w_s.size else 0)
    
    if not np.isfinite(w_min) or not np.isfinite(w_max) or abs(w_max - w_min) < 1e-12:
        w_min, w_max = 0.0, 1.0
    
    cmap = plt.cm.viridis
    norm = mpl.colors.Normalize(vmin=w_min, vmax=w_max)
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    
    def edge_colors(H):
        return [cmap(norm(H[u][v]["weight"])) for u, v in H.edges()]
    
    node_colors = [
        "tab:red" if (highlight and n == highlight) else "tab:blue"
        for n in G.nodes()
    ]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    ax1.set_title("Sparse Similarity Graph (Optimized)")
    nx.draw_networkx_nodes(G, pos, node_size=520, node_color=node_colors, ax=ax1)
    nx.draw_networkx_labels(G, pos, font_size=9, ax=ax1)
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors(G), width=2.3, alpha=0.85, ax=ax1)
    ax1.axis("off")
    fig.colorbar(sm, ax=ax1, fraction=0.046, pad=0.04, label="Edge weight = 1 − cosine similarity")
    
    ax2.set_title("Minimum Spanning Tree (Optimized)")
    nx.draw_networkx_nodes(S, pos, node_size=520, node_color=node_colors, ax=ax2)
    nx.draw_networkx_labels(S, pos, font_size=9, ax=ax2)
    nx.draw_networkx_edges(S, pos, edge_color=edge_colors(S), width=3.0, alpha=0.95, ax=ax2)
    ax2.axis("off")
    fig.colorbar(sm, ax=ax2, fraction=0.046, pad=0.04, label="Edge weight = 1 − cosine similarity")
    
    plt.tight_layout()
    plt.show()


# =============================================================================
# MAIN PIPELINE
# =============================================================================
def run_optimized_pipeline(
    text: str,
    topk: int = 5,
    minfreq: int = 1,
    maxterms: int = 200,
    visualize: bool = True,
    profile: bool = True
) -> Dict:
    """
    Optimized semantic graph construction pipeline.
    
    Complexity Analysis:
    - Stage 1 (Normalization + Term Detection): O(L + n + matches) using Aho-Corasick
    - Stage 2 (Embeddings): O(n·d) 
    - Stage 3 (Graph Building): O(n·k·log(n)) using Ball Tree ANN
    - Stage 4 (MST): O(m·log(n)) using Kruskal's with Union-Find
    - Stage 5 (Hierarchy): O(n) using Tree DP
    
    Overall: O(L + n·k·log(n)·d) vs O(n²·d) in baseline
    """
    prof = Profiler(enabled=profile)
    prof.start()
    
    # Stage 1a: Extract raw terms
    prof.begin("extract_terms")
    raw_terms = extract_terms_pattern(text)
    prof.end({"n_terms_raw": len(raw_terms)})
    
    # Stage 1b: Deduplicate
    prof.begin("dedupe_terms")
    key_to_surface = defaultdict(Counter)
    for r in raw_terms:
        k = normalize_term(r)
        if k:
            key_to_surface[k][r.strip()] += 1
    term_keys = list(key_to_surface.keys())
    prof.end({"n_terms_unique": len(term_keys)})
    
    # Stage 1c: Count occurrences (Aho-Corasick)
    prof.begin("count_terms_aho_corasick")
    stats = count_terms_optimized(text, term_keys)
    prof.end({"n_terms_found": len(stats)})
    
    # Stage 1d: Filter by frequency
    prof.begin("filter_terms")
    present = [
        (k, stats[k]["freq"], stats[k]["first_pos"]) 
        for k in stats if stats[k]["freq"] >= minfreq
    ]
    present.sort(key=lambda x: (-x[1], x[2]))
    present = present[:maxterms]
    
    if not present:
        prof.end()
        prof.stop()
        raise ValueError("No terms passed frequency filter.")
    
    kept_keys = [k for k, _, _ in present]
    labels = [
        key_to_surface[k].most_common(1)[0][0] if key_to_surface[k] else k 
        for k in kept_keys
    ]
    label_to_key = {lab: k for lab, k in zip(labels, kept_keys)}
    prof.end({"n_terms_kept": len(labels)})
    
    # Stage 2: Build embeddings
    prof.begin("build_embeddings")
    V, dim = char_ngrams_vectorized(kept_keys)
    prof.end({"embedding_dim": dim})
    
    # Stage 3: Build sparse graph
    prof.begin("build_sparse_graph")
    G, _ = build_sparse_graph_optimized(labels, V, k=topk)
    for lab in labels:
        k = label_to_key[lab]
        G.nodes[lab]["freq"] = stats[k]["freq"]
        G.nodes[lab]["first_pos"] = stats[k]["first_pos"]
    prof.end({"n_nodes": G.number_of_nodes(), "n_edges": G.number_of_edges()})
    
    # Stage 4: MST
    prof.begin("compute_mst")
    MST = minimum_spanning_tree_kruskal(G)
    prof.end({"mst_edges": MST.number_of_edges()})
    
    # Stage 5: Hierarchy (Tree DP)
    prof.begin("compute_hierarchy_tree_dp")
    hier_node, scores, total_weight = compute_hierarchy_tree_dp(MST)
    top5 = sorted(scores.items(), key=lambda x: x[1])[:5]
    prof.end()
    
    # Visualization
    prof.begin("visualize" if visualize else "skip_visualize")
    if visualize:
        visualize_graphs(G, MST, highlight=hier_node)
    prof.end()
    
    prof.stop()
    print(prof.summary())
    
    # Results
    print(f"\nDetected terms: {len(labels)}")
    print(f"Total MST weight: {total_weight:.6f}")
    print(f"Most hierarchical term: {hier_node}")
    print("Top-5 central terms:")
    for t, s in top5:
        print(f"  {t:35s}  S={s:.4f}   freq={G.nodes[t]['freq']}")
    
    return {
        "terms": labels,
        "hierarchical": hier_node,
        "scores": scores,
        "total_weight": total_weight,
        "graph": G,
        "mst": MST,
        "profiling": prof.records,
    }


# =============================================================================
# BENCHMARKING: Compare baseline vs optimized on varying sizes
# =============================================================================
def generate_synthetic_text(n_terms: int, text_length: int = 10000) -> Tuple[str, List[str]]:
    """Generate synthetic text with n_terms unique entities."""
    import random
    random.seed(42)
    
    first_names = ["John", "Jane", "Michael", "Sarah", "David", "Emily", "Robert", "Lisa"]
    last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis"]
    orgs = ["Corp", "Inc", "Labs", "Tech", "Systems", "Solutions", "Group", "Holdings"]
    
    terms = []
    for i in range(n_terms):
        if i % 3 == 0:
            name = f"{random.choice(first_names)} {random.choice(last_names)}"
        elif i % 3 == 1:
            name = f"{random.choice(last_names)} {random.choice(orgs)}"
        else:
            name = f"{''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ', k=3))}"
        terms.append(name + str(i % 100))
    
    words = ["the", "a", "is", "are", "was", "were", "has", "have", "will", "would",
             "said", "announced", "reported", "stated", "according", "to", "in", "at",
             "on", "by", "for", "with", "about", "from", "into", "through", "during"]
    
    text_parts = []
    word_count = 0
    while word_count < text_length:
        term = random.choice(terms)
        context = " ".join(random.choices(words, k=random.randint(5, 15)))
        sentence = f"{term} {context}. "
        text_parts.append(sentence)
        word_count += len(sentence.split())
    
    return " ".join(text_parts), terms


def benchmark_comparison():
    """
    Benchmark optimized vs baseline implementation on various sizes.
    """
    print("=" * 70)
    print("BENCHMARK: Optimized vs Baseline Implementation")
    print("=" * 70)
    
    # Test sizes
    sizes = [10, 25, 50, 100, 200]
    
    results = []
    
    for n in sizes:
        print(f"\n--- Testing with n={n} terms ---")
        
        text, terms = generate_synthetic_text(n, text_length=n * 50)
        
        # Run optimized
        t0 = time.perf_counter()
        try:
            result_opt = run_optimized_pipeline(
                text, topk=5, minfreq=1, maxterms=n, 
                visualize=False, profile=False
            )
            t_opt = time.perf_counter() - t0
            opt_weight = result_opt["total_weight"]
            opt_hier = result_opt["hierarchical"]
        except Exception as e:
            print(f"Optimized failed: {e}")
            t_opt = float('inf')
            opt_weight = None
            opt_hier = None
        
        # Run baseline (import from example.py)
        try:
            from example import run_pipeline
            t0 = time.perf_counter()
            result_base = run_pipeline(
                text, topk=5, minfreq=1, maxterms=n,
                visualize=False, profile=False
            )
            t_base = time.perf_counter() - t0
            base_weight = result_base["total_weight"]
            base_hier = result_base["hierarchical"]
        except Exception as e:
            print(f"Baseline failed: {e}")
            t_base = float('inf')
            base_weight = None
            base_hier = None
        
        speedup = t_base / t_opt if t_opt > 0 else float('inf')
        
        results.append({
            "n": n,
            "t_opt": t_opt,
            "t_base": t_base,
            "speedup": speedup,
            "opt_weight": opt_weight,
            "base_weight": base_weight,
            "weights_match": abs(opt_weight - base_weight) < 0.01 if opt_weight and base_weight else None
        })
        
        print(f"  Optimized: {t_opt:.4f}s, weight={opt_weight:.4f if opt_weight else 'N/A'}")
        print(f"  Baseline:  {t_base:.4f}s, weight={base_weight:.4f if base_weight else 'N/A'}")
        print(f"  Speedup:   {speedup:.2f}x")
    
    # Summary table
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"{'n terms':>10} {'Optimized':>12} {'Baseline':>12} {'Speedup':>10} {'Correct':>10}")
    print("-" * 54)
    for r in results:
        correct = "✓" if r["weights_match"] else "≈" if r["weights_match"] is None else "✗"
        print(f"{r['n']:>10} {r['t_opt']:>12.4f}s {r['t_base']:>12.4f}s {r['speedup']:>10.2f}x {correct:>10}")
    
    return results


# =============================================================================
# COMPLEXITY ANALYSIS
# =============================================================================
def print_complexity_analysis():
    """Print theoretical complexity analysis."""
    analysis = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                    COMPLEXITY ANALYSIS: OPTIMIZED VS BASELINE                 ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ Stage                    │ Baseline         │ Optimized        │ Improvement ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ 1. Text Normalization    │ O(L)             │ O(L)             │ Same        ║
║ 2. Term Detection        │ O(L·n)           │ O(L + n + m)     │ ✓ Better*   ║
║ 3. Vector Construction   │ O(n·d)           │ O(n·d)           │ Same        ║
║ 4. Similarity Graph      │ O(n²·d)          │ O(n·k·log(n)·d)  │ ✓ Much Better║
║ 5. MST Computation       │ O(m·log(n))      │ O(m·log(n))      │ Same        ║
║ 6. Hierarchy (S(u))      │ O(n²)            │ O(n)             │ ✓ Much Better║
╠══════════════════════════════════════════════════════════════════════════════╣
║ OVERALL DOMINANT TERM    │ O(n²·d)          │ O(n·k·log(n)·d)  │             ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ * Using Aho-Corasick automaton (m = number of matches)                        ║
║ * Ball Tree for approximate k-NN (k << n)                                     ║
║ * Tree DP for sum-of-distances computation                                    ║
╚══════════════════════════════════════════════════════════════════════════════╝

KEY OPTIMIZATIONS:
─────────────────

1. AHO-CORASICK FOR TERM MATCHING
   - Baseline: Compile n regex patterns, search each independently
   - Optimized: Single automaton, one pass through text
   - Improvement: O(L·n) → O(L + n + matches)

2. BALL TREE FOR APPROXIMATE NEAREST NEIGHBORS
   - Baseline: Compute full n×n similarity matrix
   - Optimized: Query k nearest neighbors per node using spatial partitioning
   - Improvement: O(n²·d) → O(n·k·log(n)·d) average case
   - Note: For k << n (sparse graphs), this is a significant improvement

3. TREE DP FOR HIERARCHY COMPUTATION
   - Baseline: All-pairs shortest paths using Dijkstra from each node
   - Optimized: Two-pass DFS exploiting tree structure
   - Improvement: O(n²) → O(n)
   - Proof: On a tree, S(u) can be computed incrementally:
     * First pass (post-order): Compute subtree sizes and internal distances
     * Second pass (pre-order): S(child) = S(parent) + w(edge) × (n - 2×subtree_size)

WHY SPARSIFICATION IS NECESSARY:
────────────────────────────────
Without sparsification, storing the full similarity graph requires O(n²) space.
For n=1000 terms, this is 1M edges, making MST computation O(n²·log(n)).
With k-sparse graphs (k≈5-20), we store only O(n·k) edges, and the MST
algorithm runs in O(n·k·log(n)) time.

WHY FREQUENCY ALONE DOESN'T CAPTURE SEMANTIC IMPORTANCE:
───────────────────────────────────────────────────────
Term frequency measures occurrence count but not semantic centrality.
A term like "the" has high frequency but low semantic importance.
The S(u) hierarchy metric captures how well-connected a term is to all
other terms in the semantic space, measuring its role as a conceptual hub.
A term with moderate frequency but central position (low S(u)) connects
disparate concepts and is thus semantically more important.
"""
    print(analysis)


# =============================================================================
# EXAMPLE USAGE
# =============================================================================
TEXT_EXAMPLE = """
Unmasking information manipulation: A quantitative approach to detecting copy-pasta, rewording, and translation on social media.

Manon Richard, Lisa Giordani, Cristian Brokate, Jean Lienard.

Abstract. This study proposes a methodology for identifying three techniques used in foreign-operated information manipulation campaigns: copy-pasta, rewording, and translation. The approach, called the "3 Delta-space duplicate methodology", quantifies three aspects of messages: semantic meaning, grapheme-level wording, and language. Pairwise distances across these dimensions help detect abnormally close messages that are likely coordinated. The method is validated using a synthetic dataset generated with ChatGPT and DeepL, and then applied to a Twitter Transparency dataset (2021) about Venezuelan actors. The method identifies all three types of inauthentic duplicates in the synthetic dataset and uncovers duplicates in the Twitter dataset across political, commercial, and entertainment contexts. The focus on clustered alterations, rather than individual messages, makes the approach efficient for large-scale detection, including AI-generated content. The method also targets translated content, which is often overlooked.

An information manipulation campaign aims to influence opinions through coordinated strategies. It typically involves identifying a target audience, studying vulnerabilities, crafting a core message, executing propagation steps, and evaluating outcomes.

A concise narrative can be transformed to reach an audience effectively. Verbatim duplication is efficient but limited by language barriers and moderation. It is also easily exposed by civil society actors, motivating more advanced techniques.

Copy-pasta, derived from "copy-paste", refers to duplicating content with minor modifications such as adding hashtags, emojis, or altering a single word. It has received limited attention in academic literature and is often detected using ad-hoc methods.

Rewording is more sophisticated and traditionally required fluent agents. With AI-generated chatbots such as ChatGPT, it has become prevalent on social networks. However, methods for distinguishing AI-generated text from benign content remain inadequate.

Translation requires reliable tools or bilingual agents to adapt messages to a different language audience. It can be costly but is important in foreign-operated campaigns. Despite its relevance, there is no established method for detecting translations within a dataset.

This study labels message pairs by duplication method using three dimensions: semantic meaning, grapheme-level wording, and language. It computes pairwise distances (Delta-semantic, Delta-grapheme, Delta-language) to identify abnormally close messages, forming the core of the "3 Delta-space duplicate methodology".

Two datasets are used. First, a synthetic dataset is generated with ChatGPT and an automatic translation tool (DeepL) to cover transformations in the 3 Delta-space. Second, the approach is applied to a Twitter dataset related to Venezuelan actors released by Twitter Transparency in 2021. The analysis identifies political boosting, commercial messages for alcoholic beverages, and entertainment messages related to "The Walking Dead". Two account typologies appear: one uses copy-pasta and rewording; the other uses rewording and translation.

Related work. Coordinated inauthentic behavior (CIB) refers to synchronized efforts to shape online discourse through repeated activity. Early work used supervised detection frameworks. More recently, unsupervised graph-based approaches have shown strong performance by computing distances between messages and clustering them.

Prior frameworks detect duplicated content with graph-based methods. Some use string-distance measures for copy-pasta; others use word-level embeddings aggregated to higher-level units. The present approach uses sentence-level representations and focuses on short texts typical of social media, aiming not only to detect duplication but to infer the duplication method and analyze network-level clusters.

The dataset used for validation has been previously studied with network analyses using user mentions and shared hashtags, identifying clusters consistent with coordinated behaviors. The present approach detects clusters and provides finer detail about the techniques used.
"""


if __name__ == "__main__":
    # Print complexity analysis
    print_complexity_analysis()
    
    # Run on example text
    print("\n" + "=" * 70)
    print("RUNNING OPTIMIZED PIPELINE ON EXAMPLE TEXT")
    print("=" * 70)
    
    result = run_optimized_pipeline(
        TEXT_EXAMPLE, 
        topk=20, 
        minfreq=1, 
        maxterms=200, 
        visualize=True, 
        profile=True
    )
    
    # Run benchmark comparison
    print("\n")
    benchmark_comparison()
