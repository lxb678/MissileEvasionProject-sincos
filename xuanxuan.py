##换电站选址代码
from __future__ import annotations
from math import radians, sin, cos, sqrt, atan2
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.termination import get_termination

# -----------------------------------------------------------------------------
# 1. RAW DATA (21 highway junctions)
# -----------------------------------------------------------------------------
pos: Dict[int, Tuple[float, float]] = {
    0: (38.60600, 109.94000),
    1: (38.46599, 110.23688),
    2: (38.75132, 109.64062),
    3: (38.23561, 109.52256),
    4: (38.02180, 109.86574),
    5: (38.01532, 109.29194),
    6: (38.16873, 110.32698),
    7: (38.37134, 109.09702),
    8: (37.75530, 110.13205),
    9: (38.72134, 109.14643),
    10: (37.71621, 109.35234),
    11: (38.65920, 110.05518),
    12: (37.93524, 109.41000),
    13: (38.43159, 109.25076),
    14: (37.76832, 109.78887),
    15: (38.84333, 110.05244),
    16: (37.22841, 108.86283),
    17: (38.23884, 110.24407),
    18: (39.07496, 109.72624),
    19: (37.97486, 110.26191),
    20: (38.96404, 109.41247),
}

# -----------------------------------------------------------------------------
# 2. HELPER FUNCTIONS
# -----------------------------------------------------------------------------

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great circle distance between two (lat, lon) points on Earth (km)."""
    R = 6371.0  # mean Earth radius in km
    φ1, φ2 = radians(lat1), radians(lat2)
    dφ = radians(lat2 - lat1)
    dλ = radians(lon2 - lon1)
    a = sin(dφ / 2) ** 2 + cos(φ1) * cos(φ2) * sin(dλ / 2) ** 2
    return 2 * R * atan2(sqrt(a), sqrt(1 - a))


def build_edges(pos: Dict[int, Tuple[float, float]], k: int = 3) -> List[Tuple[int, int, float]]:
    """Connect each node to its *k* nearest neighbours.

    Returns a list of (u, v, length_km) where u < v to avoid duplicates.
    """
    edges: set[Tuple[int, int]] = set()
    for i, (lat1, lon1) in pos.items():
        # Compute distances to every other node, sort, take k shortest
        neighbours = sorted(
            ((j, haversine_km(lat1, lon1, *pos[j])) for j in pos if j != i),
            key=lambda t: t[1],
        )[:k]
        for j, _ in neighbours:
            edges.add(tuple(sorted((i, j))))
    # Attach distance value
    return [(u, v, round(haversine_km(*pos[u], *pos[v]), 2)) for u, v in edges]

# Pre compute the edge list once so it can be reused everywhere
EDGE_LIST = build_edges(pos, k=3)  # 37 undirected edges

# -----------------------------------------------------------------------------
# 3. DEMAND MATRIX (random demo – replace with actual OD survey if available)
# -----------------------------------------------------------------------------
np.random.seed(42)  # repeatability; remove/change for new random draw
N = len(pos)
demand_matrix: np.ndarray = np.random.randint(0, 101, size=(N, N))
np.fill_diagonal(demand_matrix, 0)  # no self trips
mask = np.random.rand(N, N) < 0.20  # 20 % of OD pairs set to zero
demand_matrix[mask] = 0

# -----------------------------------------------------------------------------
# 4. NETWORK CLASS (replaces the random generator in final.py)
# -----------------------------------------------------------------------------
class RoadNetwork:
    """Fixed highway network built from external data."""

    def __init__(
        self,
        pos: Dict[int, Tuple[float, float]],
        edges: List[Tuple[int, int, float]],
        demand_matrix: np.ndarray,
    ) -> None:
        self.pos = pos
        self.num_nodes = len(pos)
        self.G = self._build_network(edges)
        self.demand_matrix = demand_matrix
        # Binary OD matrix for quick look ups (1 = demand present)
        self.od_matrix = (demand_matrix > 0).astype(int)

    # ------------------------------------------------------------------
    # PRIVATE UTILITIES
    # ------------------------------------------------------------------
    def _build_network(self, edges: List[Tuple[int, int, float]]) -> nx.Graph:
        G = nx.Graph()
        G.add_nodes_from(self.pos.keys())
        nx.set_node_attributes(G, self.pos, "pos")
        for u, v, length in edges:
            G.add_edge(u, v, length=length)
        return G

    # ------------------------------------------------------------------
    # PUBLIC METHODS
    # ------------------------------------------------------------------
    def visualize(self) -> None:
        """Side by side: network topology + OD demand heatmap."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # (a) network topology
        nx.draw(self.G, pos=self.pos, with_labels=True, node_color="lightblue", ax=ax1)
        ax1.set_title("Road Network Topology")

        # (b) OD demand heatmap
        im = ax2.imshow(self.demand_matrix, cmap="viridis")
        plt.colorbar(im, ax=ax2, label="Demand Intensity")
        ax2.set_title("OD Demand Matrix")
        ax2.set_xlabel("Destination")
        ax2.set_ylabel("Origin")
        plt.show()

# -----------------------------------------------------------------------------
# 5. OPTIMISATION PROBLEM (copied verbatim from final.py)
# -----------------------------------------------------------------------------
class BatterySwapStationProblem(ElementwiseProblem):
    def __init__(
        self,
        network: RoadNetwork,
        battery_capacity: float = 300,
        charging_rate: float = 1.0,
        safe_soc: float = 0.2,
    ) -> None:
        self.network = network
        self.battery_capacity = battery_capacity
        self.charging_rate = charging_rate
        self.safe_soc = safe_soc
        # Precompute top 3 shortest paths for every OD pair with demand
        self.k_shortest_paths = self._precompute_paths(k=3)

        n_var = network.num_nodes  # binary decision – build station or not
        super().__init__(
            n_var=n_var,
            n_obj=1,
            n_constr=0,
            xl=0,
            xu=1,
            vtype=bool,
        )

    # --------------------------------------------------------------
    # INTERNAL HELPERS
    # --------------------------------------------------------------
    # def _precompute_paths(self, k: int) -> Dict[Tuple[int, int], List[List[int]]]:
    #     paths: Dict[Tuple[int, int], List[List[int]]] = {}
    #     for o in range(self.network.num_nodes):
    #         for d in range(self.network.num_nodes):
    #             if self.network.od_matrix[o, d] == 0:
    #                 continue
    #             try:
    #                 paths[(o, d)] = list(
    #                     nx.shortest_simple_paths(self.network.G, o, d, weight="length")
    #                 )[:k]
    #             except nx.NetworkXNoPath:
    #                 paths[(o, d)] = []
    #     return paths
    # 修改后 (可以立即运行的修复版代码):
    def _precompute_paths(self, k: int) -> Dict[Tuple[int, int], List[List[int]]]:
        paths: Dict[Tuple[int, int], List[List[int]]] = {}
        print("Pre-computing single shortest paths (fast)...")  # 添加一个打印，让您知道它在工作
        for o in range(self.network.num_nodes):
            for d in range(self.network.num_nodes):
                # 跳过没有需求或起点终点相同的配对
                if self.network.od_matrix[o, d] == 0 or o == d:
                    continue
                try:
                    # 使用高效的 nx.shortest_path 函数
                    path = nx.shortest_path(self.network.G, o, d, weight="length")
                    # 将单条路径包装在列表中，以保持数据结构一致
                    paths[(o, d)] = [path]
                except nx.NetworkXNoPath:
                    paths[(o, d)] = []
        print("Finished pre-computing paths.")  # 这一步会瞬间完成
        return paths

    # --------------------------------------------------------------
    # EVALUATION
    # --------------------------------------------------------------
    def _evaluate(self, x: np.ndarray, out: Dict, *args, **kwargs):
        x_bool = x.astype(bool)
        station_nodes = np.where(x_bool)[0]

        cost = self._investment_cost(station_nodes)
        coverage = self._demand_coverage(station_nodes)

        # Single objective GA – minimise cost while rewarding coverage
        out["F"] = cost - 0.5 * coverage

    # --------------------------------------------------------------
    # COST / COVERAGE UTILITIES
    # --------------------------------------------------------------
    def _investment_cost(self, station_nodes: List[int]) -> float:
        num_stations = len(station_nodes)
        cost_per_station = (
            2.1e6  # swapping equipment
            + (0.09 + 0.01 * self.charging_rate) * self.battery_capacity * 10  # batteries
            + 0.018 * (self.battery_capacity * self.charging_rate)  # chargers
            + 0.816e6  # warehouse & land
        )
        return num_stations * cost_per_station

    def _demand_coverage(self, station_nodes: List[int]) -> float:
        total = float(np.sum(self.network.demand_matrix))
        covered = 0.0
        for (o, d), paths in self.k_shortest_paths.items():
            demand = self.network.demand_matrix[o, d]
            for path in paths:
                if self._path_is_covered(path, station_nodes):
                    covered += demand
                    break
        return covered / total * 100.0

    def _path_is_covered(self, path: List[int], station_nodes: List[int]) -> bool:
        round_trip = path + path[-2::-1]  # O→…→D→…→O
        energy = 0.0
        for i in range(len(round_trip) - 1):
            u, v = round_trip[i], round_trip[i + 1]
            segment = self.network.G.edges[u, v]["length"]
            energy += segment * 1.7  # 1.7 kWh per km
            if v in station_nodes:
                # need to arrive with ≥ safe_soc
                if energy > self.battery_capacity * (1 - self.safe_soc):
                    return False
                energy = 0.0  # reset after swap
        return energy <= self.battery_capacity * self.safe_soc

# -----------------------------------------------------------------------------
# 6. GA DRIVER (unchanged)
# -----------------------------------------------------------------------------

def solve_with_ga(
    problem: BatterySwapStationProblem,
    pop_size: int = 10,
    n_gen: int = 15,
):
    algorithm = GA(
        pop_size=pop_size,
        sampling=BinaryRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(prob=0.1, eta=20),
        eliminate_duplicates=True,
    )
    termination = get_termination("n_gen", n_gen)
    result = minimize(problem, algorithm, termination, seed=42, verbose=True)
    return result

# -----------------------------------------------------------------------------
# 7. VISUALISATION
# -----------------------------------------------------------------------------

def visualize_solution(
    network: RoadNetwork,
    solution: np.ndarray,
    problem: BatterySwapStationProblem,
):
    plt.figure(figsize=(10, 8))
    pos = network.pos
    nx.draw_networkx_nodes(network.G, pos, node_color="lightblue", label="Candidate Nodes")
    nx.draw_networkx_edges(network.G, pos, alpha=0.3)

    station_nodes = np.where(solution)[0]
    nx.draw_networkx_nodes(
        network.G,
        pos,
        nodelist=station_nodes.tolist(),
        node_color="red",
        label="Selected Stations",
    )

    # highlight high demand (>50 trips) OD pairs
    high_od = np.argwhere(network.demand_matrix > 50)
    for o, d in high_od:
        if (o, d) in problem.k_shortest_paths and problem.k_shortest_paths[(o, d)]:
            path = problem.k_shortest_paths[(o, d)][0]
            edges = list(zip(path[:-1], path[1:]))
            nx.draw_networkx_edges(network.G, pos, edgelist=edges, edge_color="green", width=2)

    plt.legend()
    coverage_pct = problem._demand_coverage(station_nodes)
    plt.title(f"Optimal BSS Configuration (coverage = {coverage_pct:.1f}% )")
    plt.show()

# -----------------------------------------------------------------------------
# 8. MAIN ENTRY POINT
# -----------------------------------------------------------------------------

def main():
    # Build fixed network instance
    network = RoadNetwork(pos, EDGE_LIST, demand_matrix)
    network.visualize()

    # GA optimisation
    problem = BatterySwapStationProblem(
        network,
        battery_capacity=350,  # kWh per pack
        charging_rate=1.5,     # C rate
        safe_soc=0.2,          # 20 % minimum
    )
    print("\nRunning genetic algorithm …")
    result = solve_with_ga(problem, pop_size=30, n_gen=50)

    # Visualise + print key numbers
    visualize_solution(network, result.X, problem)
    station_nodes = np.where(result.X)[0]
    print("\n=== Optimisation Results ===")
    print(f"Selected nodes      : {station_nodes}")
    print(f"Number of stations  : {len(station_nodes)}")
    print(f"Total investment    : {problem._investment_cost(station_nodes) / 1e6:.2f} M CNY")
    print(f"Demand coverage     : {problem._demand_coverage(station_nodes):.1f}%")


if __name__ == "__main__":
    main()

