# Semantic Graph from Text via NER (if available) or Pattern Matching
# + edge-weight heatmap visualization (NetworkX)
# + runtime + memory profiling (per-stage + total)
#
# Usage in Colab:
#   1) Put your text in TEXT (or load from file)
#   2) Run run_pipeline(TEXT, topk=5, minfreq=2, maxterms=60, visualize=True)

import re, math, unicodedata, time
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl

# ---- Memory profiling (works in Colab/Linux). Falls back gracefully otherwise.
try:
    import tracemalloc
except Exception:
    tracemalloc = None

try:
    import resource  # Unix
except Exception:
    resource = None


# -----------------------------
# Profiling helpers
# -----------------------------
def _bytes_to_mb(x: int) -> float:
    return float(x) / (1024.0 * 1024.0)

def _rss_mb() -> Optional[float]:
    """Resident set size in MB (best-effort)."""
    if resource is None:
        return None
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux: ru_maxrss is KB; macOS: bytes. Colab is Linux.
    if ru > 10_000_000:  # heuristic for bytes scale
        return _bytes_to_mb(ru)
    return ru / 1024.0  # KB -> MB

class StageProfiler:
    """
    Tracks wall time and memory deltas per stage.
    - Wall time: time.perf_counter
    - Memory: tracemalloc peak (Python allocations) + RSS (process, best-effort)
    """
    def __init__(self, enabled: bool = True):
        self.enabled = enabled and (tracemalloc is not None)
        self.rows = []  # list of dicts
        self._t0 = None
        self._snap0 = None
        self._rss0 = None

    def start(self):
        if self.enabled:
            tracemalloc.start()
        self._rss0 = _rss_mb()

    def stop(self):
        if self.enabled:
            tracemalloc.stop()

    def begin(self, name: str):
        if not self.enabled:
            self._t0 = time.perf_counter()
            self._rss0_stage = _rss_mb()
            self._name = name
            return
        self._name = name
        self._t0 = time.perf_counter()
        self._snap0 = tracemalloc.take_snapshot()
        self._rss0_stage = _rss_mb()

    def end(self, extra: Optional[Dict] = None):
        t1 = time.perf_counter()
        dt = t1 - (self._t0 or t1)

        rss1 = _rss_mb()
        rss_delta = None if (rss1 is None or self._rss0_stage is None) else (rss1 - self._rss0_stage)

        if self.enabled:
            snap1 = tracemalloc.take_snapshot()
            stats = snap1.compare_to(self._snap0, "lineno")
            py_alloc = sum(s.size_diff for s in stats)  # bytes diff (net)
            current, peak = tracemalloc.get_traced_memory()
            row = {
                "stage": self._name,
                "time_s": dt,
                "py_alloc_net_mb": _bytes_to_mb(int(py_alloc)),
                "py_peak_mb": _bytes_to_mb(int(peak)),
                "rss_delta_mb": rss_delta,
                "rss_mb": rss1,
            }
        else:
            row = {
                "stage": self._name,
                "time_s": dt,
                "py_alloc_net_mb": None,
                "py_peak_mb": None,
                "rss_delta_mb": rss_delta,
                "rss_mb": rss1,
            }

        if extra:
            row.update(extra)
        self.rows.append(row)

    def summary(self):
        # Pretty print
        print("\n=== Profiling summary (wall time + memory) ===")
        for r in self.rows:
            parts = [
                f"{r['stage']}: {r['time_s']:.4f}s",
            ]
            if r["py_peak_mb"] is not None:
                parts.append(f"py_peak={r['py_peak_mb']:.2f}MB")
                parts.append(f"py_net={r['py_alloc_net_mb']:.2f}MB")
            if r["rss_mb"] is not None:
                parts.append(f"rss={r['rss_mb']:.2f}MB")
                if r["rss_delta_mb"] is not None:
                    parts.append(f"rss_delta={r['rss_delta_mb']:+.2f}MB")
            # optional extras
            for k in ("n_terms_raw", "n_terms_kept", "n_nodes", "n_edges", "n_span_edges", "tfidf_dim"):
                if k in r:
                    parts.append(f"{k}={r[k]}")
            print("  - " + " | ".join(parts))


# -----------------------------
# Normalization utilities
# -----------------------------
def strip_accents(s: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch))

def normalize_for_matching(s: str) -> str:
    s = strip_accents(s).lower()
    s = re.sub(r"[^a-z0-9\s\-]", " ", s)   # keep letters/digits/spaces/hyphens
    s = re.sub(r"\s+", " ", s).strip()
    return s

def normalize_term_key(s: str) -> str:
    s = normalize_for_matching(s)
    s = s.replace("-", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


# -----------------------------
# Term extraction (NER preferred; fallback pattern matching)
# -----------------------------
def extract_terms_spacy_ner(text: str) -> Optional[List[str]]:
    try:
        import spacy  # type: ignore
    except Exception:
        return None

    nlp = None
    for model in ("en_core_web_sm", "en_core_web_md", "en_core_web_lg"):
        try:
            nlp = spacy.load(model)
            break
        except Exception:
            pass
    if nlp is None:
        return None

    doc = nlp(text)
    keep = {"PERSON", "ORG", "GPE", "LOC", "PRODUCT", "EVENT", "WORK_OF_ART"}
    ents = [ent.text.strip() for ent in doc.ents if ent.label_ in keep and ent.text.strip()]
    return ents if ents else None

def extract_terms_pattern(text: str) -> List[str]:
    tokens = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]+(?:[-'][A-Za-zÀ-ÖØ-öø-ÿ0-9]+)?", text)
    connectors = {
        "of","the","and","de","del","la","las","los","da","do","dos","das","di",
        "van","von","y","&"
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
                    phrase.append(t2); j += 1; continue
                if is_caps(t2):
                    phrase.append(t2); j += 1; continue
                break
            while phrase and phrase[-1].lower() in connectors:
                phrase.pop()
            phrases.append(" ".join(phrase))
            i = j
        else:
            i += 1

    phrases.extend(re.findall(r"\b[A-Z]{2,}\b", text))
    return phrases

def extract_terms(text: str) -> List[str]:
    ner = extract_terms_spacy_ner(text)
    return ner if ner is not None else extract_terms_pattern(text)


# -----------------------------
# Counting occurrences in text (pattern matching)
# -----------------------------
def build_term_patterns(term_keys: List[str]) -> Dict[str, re.Pattern]:
    patterns = {}
    for key in term_keys:
        toks = key.split()
        if not toks:
            continue
        mid = r"(?:\s+|-)"  # treat spaces/hyphens as equivalent
        core = mid.join(re.escape(t) for t in toks)
        patterns[key] = re.compile(rf"\b{core}\b", flags=re.IGNORECASE)
    return patterns

def count_terms_in_text(text: str, term_keys: List[str], patterns: Dict[str, re.Pattern]) -> Dict[str, Dict]:
    text_norm = normalize_for_matching(text)
    out = {}
    for key in term_keys:
        rx = patterns[key]
        matches = list(rx.finditer(text_norm))
        if not matches:
            continue
        freq = len(matches)
        first_pos = matches[0].start()
        ctxs = []
        for m in matches[:2]:
            a = max(0, m.start() - 40)
            b = min(len(text_norm), m.end() + 40)
            ctxs.append(text_norm[a:b])
        out[key] = {"freq": freq, "first_pos": first_pos, "contexts": ctxs}
    return out


# -----------------------------
# Character 3-gram TF-IDF embeddings (offline baseline)
# -----------------------------
def char_ngrams(s: str, n: int = 3) -> List[str]:
    s = s.replace(" ", "_")
    if len(s) < n:
        return [s] if s else []
    return [s[i:i+n] for i in range(len(s) - n + 1)]

def build_tfidf_char3(term_keys: List[str]) -> Tuple[np.ndarray, int]:
    N = len(term_keys)
    counts = [Counter(char_ngrams(t, 3)) for t in term_keys]
    df = Counter()
    for c in counts:
        for g in c:
            df[g] += 1

    feats = sorted(df.keys())
    idx = {g:i for i,g in enumerate(feats)}
    V = np.zeros((N, len(feats)), dtype=np.float64)

    idf = {g: math.log((N + 1) / (df[g] + 1)) + 1.0 for g in feats}
    for i, c in enumerate(counts):
        for g, tf in c.items():
            V[i, idx[g]] = tf * idf[g]

    norms = np.linalg.norm(V, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (V / norms), len(feats)


# -----------------------------
# Graph building + connectivity + spanning + hierarchy
# -----------------------------
def build_sparse_graph(labels: List[str], V: np.ndarray, topk: int) -> Tuple[nx.Graph, np.ndarray]:
    n = len(labels)
    sim = V @ V.T
    np.fill_diagonal(sim, -1.0)

    G = nx.Graph()
    G.add_nodes_from(labels)

    if n <= 1:
        return G, sim

    kk = max(1, min(topk, n - 1))
    for i in range(n):
        nbrs = np.argpartition(-sim[i], kk)[:kk]
        nbrs = nbrs[np.argsort(-sim[i][nbrs])]
        for j in nbrs:
            if j == i:
                continue
            w = 1.0 - float(sim[i, j])
            u, v = labels[i], labels[j]
            if G.has_edge(u, v):
                if w < G[u][v]["weight"]:
                    G[u][v]["weight"] = w
                    G[u][v]["sim"] = float(sim[i, j])
            else:
                G.add_edge(u, v, weight=w, sim=float(sim[i, j]))

    comps = list(nx.connected_components(G))
    label_to_idx = {lab:i for i,lab in enumerate(labels)}
    while len(comps) > 1:
        best = None  # (w,u,v,s)
        comp_list = [list(c) for c in comps]
        for a in range(len(comp_list)):
            Ai = np.array([label_to_idx[x] for x in comp_list[a]], dtype=int)
            for b in range(a+1, len(comp_list)):
                Bi = np.array([label_to_idx[x] for x in comp_list[b]], dtype=int)
                block = sim[np.ix_(Ai, Bi)]
                flat = int(np.argmax(block))
                ia = flat // block.shape[1]
                ib = flat % block.shape[1]
                smax = float(block[ia, ib])
                w = 1.0 - smax
                u = labels[Ai[ia]]
                v = labels[Bi[ib]]
                if best is None or w < best[0]:
                    best = (w, u, v, smax)
        if best is None:
            break
        w, u, v, s = best
        G.add_edge(u, v, weight=w, sim=s)
        comps = list(nx.connected_components(G))

    return G, sim

def spanning_structure_min_weight(G: nx.Graph) -> nx.Graph:
    return nx.minimum_spanning_tree(G, weight="weight")

def hierarchy_in_tree(tree: nx.Graph) -> Tuple[str, Dict[str, float]]:
    scores = {}
    for u, dist_map in nx.all_pairs_dijkstra_path_length(tree, weight="weight"):
        scores[u] = float(sum(dist_map.values()))
    best = min(scores, key=scores.get)
    return best, scores


# -----------------------------
# Visualization with heatmap edge weights
# -----------------------------
def draw_heatmap_graphs(G: nx.Graph, S: nx.Graph, highlight: Optional[str] = None):
    pos = nx.spring_layout(G, seed=7)

    def weights(H: nx.Graph) -> np.ndarray:
        return np.array([H[u][v]["weight"] for u,v in H.edges()], dtype=float) if H.number_of_edges() else np.array([])

    w_all = weights(G)
    w_span = weights(S)

    if w_all.size == 0 and w_span.size == 0:
        print("Nothing to draw (no edges).")
        return

    w_min = float(min(w_all.min() if w_all.size else float("inf"),
                      w_span.min() if w_span.size else float("inf")))
    w_max = float(max(w_all.max() if w_all.size else float("-inf"),
                      w_span.max() if w_span.size else float("-inf")))
    if not np.isfinite(w_min) or not np.isfinite(w_max) or abs(w_max - w_min) < 1e-12:
        w_min, w_max = 0.0, 1.0

    cmap = plt.cm.viridis
    norm = mpl.colors.Normalize(vmin=w_min, vmax=w_max)
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    def edge_colors(H: nx.Graph):
        return [cmap(norm(H[u][v]["weight"])) for u,v in H.edges()]

    node_colors = [
        "tab:red" if (highlight is not None and n == highlight) else "tab:blue"
        for n in G.nodes()
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.set_title("Sparse Similarity Graph (edge weight heatmap)")
    nx.draw_networkx_nodes(G, pos, node_size=520, node_color=node_colors, ax=ax1)
    nx.draw_networkx_labels(G, pos, font_size=9, ax=ax1)
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors(G), width=2.3, alpha=0.85, ax=ax1)
    ax1.axis("off")
    fig.colorbar(sm, ax=ax1, fraction=0.046, pad=0.04,
                 label="Edge weight = 1 − cosine similarity")

    ax2.set_title("Minimum-Weight Spanning Structure (edge weight heatmap)")
    nx.draw_networkx_nodes(S, pos, node_size=520, node_color=node_colors, ax=ax2)
    nx.draw_networkx_labels(S, pos, font_size=9, ax=ax2)
    nx.draw_networkx_edges(S, pos, edge_color=edge_colors(S), width=3.0, alpha=0.95, ax=ax2)
    ax2.axis("off")
    fig.colorbar(sm, ax=ax2, fraction=0.046, pad=0.04,
                 label="Edge weight = 1 − cosine similarity")

    plt.tight_layout()
    plt.show()


# -----------------------------
# End-to-end function (with profiling)
# -----------------------------
def run_pipeline(
    text: str,
    topk: int = 5,
    minfreq: int = 2,
    maxterms: int = 60,
    visualize: bool = True,
    profile: bool = True,
):
    prof = StageProfiler(enabled=profile)
    prof.start()

    prof.begin("extract_terms")
    raw_terms = extract_terms(text)
    prof.end({"n_terms_raw": len(raw_terms)})

    prof.begin("dedupe_terms")
    key_to_surface = defaultdict(Counter)
    for r in raw_terms:
        k = normalize_term_key(r)
        if k:
            key_to_surface[k][r.strip()] += 1
    term_keys = list(key_to_surface.keys())
    prof.end({"n_terms_kept": len(term_keys)})

    prof.begin("build_patterns+count")
    patterns = build_term_patterns(term_keys)
    stats = count_terms_in_text(text, term_keys, patterns)
    prof.end({"n_terms_kept": len(stats)})

    prof.begin("filter_terms")
    present = [(k, stats[k]["freq"], stats[k]["first_pos"]) for k in stats if stats[k]["freq"] >= minfreq]
    present.sort(key=lambda x: (-x[1], x[2]))
    present = present[:maxterms]
    if not present:
        prof.end()
        prof.stop()
        raise ValueError("No extracted terms passed minfreq. Lower minfreq or use longer text.")
    kept_keys = [k for k,_,_ in present]
    labels = [key_to_surface[k].most_common(1)[0][0] if key_to_surface[k] else k for k in kept_keys]
    label_to_key = {lab:k for lab,k in zip(labels, kept_keys)}
    prof.end({"n_terms_kept": len(labels)})

    prof.begin("tfidf_embeddings")
    V, tfidf_dim = build_tfidf_char3(kept_keys)
    prof.end({"tfidf_dim": tfidf_dim})

    prof.begin("build_graph")
    G, _ = build_sparse_graph(labels, V, topk=topk)
    # Attach node metadata
    for lab in labels:
        k = label_to_key[lab]
        G.nodes[lab]["freq"] = stats[k]["freq"]
        G.nodes[lab]["first_pos"] = stats[k]["first_pos"]
    prof.end({"n_nodes": G.number_of_nodes(), "n_edges": G.number_of_edges()})

    prof.begin("spanning_structure")
    S = spanning_structure_min_weight(G)
    prof.end({"n_span_edges": S.number_of_edges()})

    prof.begin("hierarchy")
    hier, scores = hierarchy_in_tree(S)
    total_weight = float(sum(S[u][v]["weight"] for u,v in S.edges()))
    top5 = sorted(scores.items(), key=lambda x: x[1])[:5]
    prof.end()

    prof.begin("visualize" if visualize else "visualize(skipped)")
    if visualize:
        draw_heatmap_graphs(G, S, highlight=hier)
    prof.end()

    prof.stop()
    prof.summary()

    print("\nDetected terms used:", len(labels))
    print("Total weight (spanning structure):", total_weight)
    print("Most hierarchical term:", hier)
    print("Top-5 central terms:")
    for t, s in top5:
        print(f"  {t:35s}  S={s:.4f}   freq={G.nodes[t]['freq']}")

    return {
        "terms": labels,
        "node_stats": {lab: {"freq": G.nodes[lab]["freq"], "first_pos": G.nodes[lab]["first_pos"]} for lab in labels},
        "edges": list(G.edges(data=True)),
        "span_edges": list(S.edges(data=True)),
        "hierarchical": hier,
        "scores": scores,
        "total_weight": total_weight,
        "profiling": prof.rows,
    }


# -----------------------------
# Minimal Colab usage snippet
# -----------------------------
TEXT = """
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

# Run (profiling on by default)
result = run_pipeline(TEXT, topk=20, minfreq=1, maxterms=200, visualize=True, profile=True)
