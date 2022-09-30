from copy import deepcopy
import itertools
import pathlib
import random
from typing import Callable, Dict, Generator, Hashable, List, Optional, Tuple

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.animation import FuncAnimation

thisdir = pathlib.Path(__file__).parent.resolve()

class Polygon:
    def __init__(self, points: List[Tuple[float, float]]) -> None:
        self.points = points

    def _distance(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    @property
    def _distances(self) -> List[float]:
        return [self._distance(p1, p2) for p1, p2 in zip(self.points, self.points[1:] + [self.points[0]])]

    @property
    def perimeter(self) -> float:
        return sum(self._distances)

    @property
    def _cum_distances(self) -> List[float]:
        return np.cumsum(self._distances)

    def _get_point(self, distance: float) -> Tuple[float, float]:
        if distance < 0:
            raise ValueError("Distance must be positive")
            
        distance = distance % self.perimeter

        index = np.searchsorted(self._cum_distances, distance)
        cum_distance = 0 if index == 0 else self._cum_distances[index-1]

        p1 = self.points[index]
        p2 = self.points[(index+1) % len(self.points)]

        d = distance - cum_distance
        return (p1[0] + ((p2[0] - p1[0]) / self._distances[index] * d),
                p1[1] + ((p2[1] - p1[1]) / self._distances[index] * d))

def to_fully_connected(graph: nx.Graph) -> nx.Graph:
    graph = graph.copy()
    for node1, node2 in nx.non_edges(graph):
        bandwidth = 1 / nx.shortest_path_length(
            graph, node1, node2, 
            weight=lambda u, v, d: 1/d["bandwidth"]
        )
        graph.add_edge(node1, node2, bandwidth=bandwidth)
    return graph

def run_workflow(workflow: nx.DiGraph, 
                 network: nx.Graph, 
                 schedule: Dict[Hashable, Hashable],
                 finish_times: Optional[Dict[Hashable, float]] = None,
                 stop_time: Optional[float] = None) -> Tuple[Dict[Hashable, float], bool]:
    if finish_times is None:
        finish_times = {}
    for task in nx.topological_sort(workflow):
        if task not in schedule:
            raise ValueError(f"Task {task} not scheduled")
        if task in finish_times:
            continue # already calculated
        parent_tasks = list(workflow.predecessors(task))
        comm_time = max(
            (
                0 if schedule[parent_task] == schedule[task] else
                workflow.edges[parent_task, task]['data']/network.edges[schedule[parent_task], schedule[task]]["bandwidth"]
            )
            for parent_task in parent_tasks
        ) if parent_tasks else 0
        # print("Comm Time for", task, "is", comm_time)
        # print("Exec time for", task, "is", workflow.nodes[task]["cost"] / network.nodes[schedule[task]]["cpu"])
        start_time = max(
            finish_times[parent_task] + (
                0 if schedule[parent_task] == schedule[task] else
                workflow.edges[parent_task, task]['data'] / network.edges[schedule[parent_task], schedule[task]]["bandwidth"]
            )
            for parent_task in parent_tasks
        ) if parent_tasks else 0
        finish_time = start_time + workflow.nodes[task]["cost"] / network.nodes[schedule[task]]["cpu"]
        if stop_time is not None and finish_time > stop_time:
            return finish_times, False
        finish_times[task] = finish_time
    return finish_times, True
        
class Simulation:
    def __init__(self,
                 polygon: Polygon, 
                 agent_cpus: int,
                 timestep: float,
                 radius_threshold: float,
                 bandwidth: Callable[[float], float]) -> None:
        self.polygon = polygon
        self.agent_cpus = agent_cpus
        self.num_agents = len(agent_cpus)
        self.timestep = timestep
        self.radius_threshold = radius_threshold
        self.bandwidth = bandwidth

    def init_positions(self) -> List[Tuple[float, float]]:
        dists = np.arange(0, self.polygon.perimeter, self.polygon.perimeter / self.num_agents)
        return [self.polygon._get_point(d) for d in dists]

    def run(self, 
            rounds: Optional[int] = None,
            get_workflow: Optional[Callable[[], nx.DiGraph]] = None,
            scheduler: Optional[Callable[[nx.DiGraph, nx.Graph], Dict[Hashable, Hashable]]] = None) -> Generator[nx.Graph, None, None]:
        i = 0
        if get_workflow is not None:
            workflow = get_workflow()
        schedule: Dict[Hashable, Hashable] = {}
        finish_times: Dict[Hashable, float] = {}
        sim_time = 0
        while True:
            sim_time = i * self.timestep

            # move Agents and get comm network
            dists = (np.arange(0, self.polygon.perimeter, self.polygon.perimeter / self.num_agents) + sim_time) % self.polygon.perimeter
            pos = [self.polygon._get_point(d) for d in dists]
            network = nx.Graph()
            network.add_nodes_from([
                (node, {"pos": p, "cpu": self.agent_cpus[node]}) 
                for node, p in zip(range(self.num_agents), pos)
            ])
            for src, dst in nx.geometric_edges(network, radius=self.radius_threshold):
                d = np.sqrt((pos[src][0] - pos[dst][0]) ** 2 + (pos[src][1] - pos[dst][1]) ** 2)
                network.add_edge(src, dst, bandwidth=self.bandwidth(d))
            yield network
            i += 1

            # Execute workflow assuming network is valid for next timestep time
            if get_workflow is not None and scheduler is not None:
                schedule = scheduler(workflow, network)
                finish_times, success = run_workflow(
                    workflow, network, schedule, finish_times, sim_time+self.timestep
                )
                if success:
                    makespan = max(finish_times.values())
                    print(f"Finished in {makespan:.2f} seconds")
                    workflow = get_workflow()

            # Stop if we have reached the end
            if rounds is not None and i >= rounds:
                break


def visual_sim(sim: Simulation) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.tight_layout()
    ax1.axis('equal')
    ax2.axis('equal')

    ax1.add_patch(patches.Polygon(sim.polygon.points, fill=False, edgecolor="black"))
    
    sim_run = sim.run()
    init_network = next(sim_run)
    xdata, ydata = list(zip(*[pos for _, pos in nx.get_node_attributes(init_network, "pos").items()]))
    ln, = ax1.plot(xdata, ydata, 'ro')

    # draw network on ax2
    network_pos = nx.circular_layout(init_network)
    nx.draw(init_network, network_pos, with_labels=True, ax=ax2)

    round = 0
    def update(frame):
        nonlocal round
        round += 1
        print(f"Round {round}")
        network = next(sim_run)
        xdata, ydata = list(zip(*[pos for _, pos in nx.get_node_attributes(network, "pos").items()]))
        ln.set_data(xdata, ydata)

        ax2.clear()
        nx.draw(network, network_pos, with_labels=True, ax=ax2)
        return ln,

    ani = FuncAnimation(fig, update, frames=100, interval=100)
    plt.show()

def save_sim_networks(sim: Simulation, savedir: pathlib.Path, rounds: int = 100) -> None:
    savedir.mkdir(parents=True, exist_ok=True)
    for i, network in enumerate(sim.run(rounds=rounds)):
        nx.write_gml(to_fully_connected(network), savedir / f"network_{i}.gml")

def main():
    polygon = Polygon([(0, 0), (10, 0), (10, 10), (5, 10), (5, 5), (2, 5), (2, 10), (0, 10)])
    sim = Simulation(
        polygon=polygon,
        agent_cpus=[random.random() for _ in range(10)],
        timestep=0.1,
        radius_threshold=7,
        bandwidth=lambda x: 1 / (1 + np.exp(-x))
    )

    visual_sim(sim)
    # save_sim_networks(sim, pathlib.Path("sim_networks"))

        

if __name__ == '__main__':
    main()
