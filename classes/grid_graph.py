import networkx as nx
import numpy as np
from typing import Tuple, Dict


class reachable_graph:
    def __init__(self, reachable_points: list, step_size: float = 0.25, tol: float = 1e-9):
        self.step_size = float(step_size)
        self.points = [tuple(p) for p in reachable_points]
        self.tol = float(tol)

        self.graph: nx.Graph = None
        self.node_coords: np.ndarray = None
        self.node_coords_dict: Dict[str, Tuple[float, float]] = None

        self.graph2: nx.Graph = None
        self.node_coords2: np.ndarray = None
        self.node_coords2_dict: Dict[str, Tuple[float, float]] = None

    def _round_pt(self, p: Tuple[float, float]) -> Tuple[float, float]:
        t = self.tol
        return (round(p[0] / t) * t, round(p[1] / t) * t)

    def build_graph(self) -> tuple:
        G = nx.Graph()
        coords = {str(i): self._round_pt(tuple(pos)) for i, pos in enumerate(self.points)}
        idx_by_coord = {c: n for n, c in coords.items()}

        step = self.step_size
        directions = [(step, 0.0), (-step, 0.0), (0.0, step), (0.0, -step)]

        for n, (x, y) in coords.items():
            G.add_node(n)
            for dx, dy in directions:
                nbr = self._round_pt((x + dx, y + dy))
                m = idx_by_coord.get(nbr)
                if m is not None and m != n:
                    G.add_edge(n, m, weight=step, length=step)

        self.graph = G
        self.node_coords_dict = coords
        self.node_coords = np.array([coords[n] for n in coords])

        return self.graph, self.node_coords

    def build_pruned_graph(self, available_poses: list) -> tuple:
        if self.graph is None or self.node_coords_dict is None:
            raise RuntimeError("Call build_graph() before build_pruned_graph().")

        avail_set = {self._round_pt(tuple(p)) for p in available_poses}
        keep_nodes = [n for n, c in self.node_coords_dict.items() if c in avail_set]
        G2 = self.graph.subgraph(keep_nodes).copy()
        coords2 = {n: self.node_coords_dict[n] for n in keep_nodes}
        # print("Working")
        self.graph2 = G2
        self.node_coords2_dict = coords2
        self.node_coords2 = np.array([coords2[n] for n in coords2])

        return self.graph2, self.node_coords2

    def find_coord_idx(self, coord: np.ndarray) -> int:
        if self.node_coords is None:
            raise RuntimeError("Call build_graph() before find_coord_idx().")
        dists = np.linalg.norm(self.node_coords - coord, axis=1)
        return int(np.argmin(dists))