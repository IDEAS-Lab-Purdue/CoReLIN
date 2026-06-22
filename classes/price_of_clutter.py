from typing import Dict, Hashable, Iterable, Optional, Tuple, Union, List
import math
import statistics
import networkx as nx

Coord = Tuple[float, float]
Coords = Dict[Hashable, Coord]

def poc(
    graph_base: Union[nx.Graph, nx.DiGraph],
    graph_clutter: Union[nx.Graph, nx.DiGraph],
    nc_base: Coords,
    nc_clutter: Coords,
    weight: str = "weight",
    pairs: Optional[Iterable[Tuple[Hashable, Hashable]]] = None,   # NEW: explicit O–D pairs
    pair_weights: Optional[Iterable[float]] = None,                 # NEW: optional weights per pair
    od_nodes: Optional[Iterable[Hashable]] = None,
    cap: Optional[float] = None,
    return_details: bool = False,
) -> Union[float, Tuple[float, dict]]:
    """
    Compute the Price of Clutter (PoC) over either:
      (a) a user-specified list of O-D pairs (optional weights), or
      (b) all ordered pairs among a node subset (od_nodes), defaulting to all baseline nodes.

    - Baseline pairs with infinite distance are excluded from both numerator and denominator.
    - For each included pair, the cluttered distance is used; if endpoint missing or path
      is infinite, the 'cap' penalty is used.
    """

    # ---- helpers ------------------------------------------------------------
    def euclid_len(u, v, coords):
        try:
            x1, y1 = coords[u]
            x2, y2 = coords[v]
            return math.hypot(x1 - x2, y1 - y2)
        except Exception:
            return 1.0  # last-resort fallback if coords missing

    def make_weight_func(G, coords):
        def wf(u, v, edict):
            w = edict.get(weight, None)
            if w is not None:
                return float(w)
            return euclid_len(u, v, coords)
        return wf

    def multi_source_sssp(G, coords, sources: Iterable[Hashable]) -> Dict[Hashable, Dict[Hashable, float]]:
        """Run Dijkstra once per unique source; return {s: {t: dist}}."""
        wf = make_weight_func(G, coords)
        out = {}
        for s in sources:
            if s in G:
                out[s] = nx.single_source_dijkstra_path_length(G, s, weight=wf)
            else:
                out[s] = {}
        return out

    # ---- choose OD pairs ----------------------------------------------------
    using_explicit_pairs = pairs is not None
    if using_explicit_pairs:
        pairs = [(s, t) for (s, t) in pairs if s != t]
        if not pairs:
            raise ValueError("Provided 'pairs' is empty or only contains self-pairs.")
        if pair_weights is not None:
            pair_weights = list(pair_weights)
            if len(pair_weights) != len(pairs):
                raise ValueError("Length of 'pair_weights' must match length of 'pairs'.")
        else:
            pair_weights = [1.0] * len(pairs)
        # sources we need to solve from (baseline & clutter)
        base_sources = {s for s, _ in pairs}
        clutter_sources = base_sources  # same sources; missing nodes handled later
    else:
        # Node subset (default: all nodes in baseline graph)
        if od_nodes is None:
            od_nodes = list(graph_base.nodes())
        else:
            od_nodes = [n for n in od_nodes if n in graph_base]
        if len(od_nodes) <= 1:
            raise ValueError("OD set has ≤1 node. Provide more nodes or let the function use all baseline nodes.")
        pairs = [(s, t) for s in od_nodes for t in od_nodes if s != t]
        pair_weights = [1.0] * len(pairs)
        base_sources = set(od_nodes)
        clutter_sources = {n for n in od_nodes if n in graph_clutter}

    # ---- baseline distances -------------------------------------------------
    d_base_all = multi_source_sssp(graph_base, nc_base, base_sources)

    # Keep only pairs with finite baseline distance
    baseline_pairs: List[Tuple[Hashable, Hashable]] = []
    baseline_dvals: List[float] = []
    kept_weights: List[float] = []

    for (s, t), w in zip(pairs, pair_weights):
        d_b = d_base_all.get(s, {}).get(t, math.inf)
        if math.isfinite(d_b):
            baseline_pairs.append((s, t))
            baseline_dvals.append(d_b)
            kept_weights.append(float(w))

    if not baseline_pairs:
        raise ValueError("No finite baseline O-D pairs in the provided set (after filtering).")

    # ---- cap from baseline median if needed --------------------------------
    if cap is None:
        med = statistics.median(baseline_dvals)
        cap = 10.0 * med if med > 0 else 10.0

    # ---- clutter distances --------------------------------------------------
    # Only run SSSP for sources that exist in the clutter graph
    d_clutter_all = multi_source_sssp(graph_clutter, nc_clutter, clutter_sources)

    numerator = 0.0
    denominator = 0.0
    num_infinite = 0
    num_endpoints_pruned = 0

    # For quick membership check of endpoints in clutter graph
    clutter_nodes = graph_clutter.nodes

    for (s, t), d_b, w in zip(baseline_pairs, baseline_dvals, kept_weights):
        denominator += w * d_b

        # If either endpoint absent in clutter graph, count as cap
        if s not in clutter_nodes or t not in clutter_nodes:
            numerator += w * cap
            num_endpoints_pruned += 1
            continue

        d_c = d_clutter_all.get(s, {}).get(t, math.inf)
        if math.isfinite(d_c):
            numerator += w * min(d_c, cap)
        else:
            numerator += w * cap
            num_infinite += 1

    poc_value = numerator / denominator

    if return_details:
        details = {
            "cap": cap,
            "num_pairs": len(baseline_pairs),
            "denominator": denominator,
            "numerator": numerator,
            "num_infinite_in_clutter": num_infinite,
            "num_endpoints_pruned": num_endpoints_pruned,
            "baseline_dist_stats": {
                "median": statistics.median(baseline_dvals),
                "mean": statistics.fmean(baseline_dvals),
            },
            "weighted": (pairs is not None and any(w != 1.0 for w in kept_weights)),
        }
        return poc_value, details

    return poc_value