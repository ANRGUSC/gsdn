import pathlib
from typing import Callable, Dict, Generator, Hashable, List, Optional, Tuple

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
from matplotlib.animation import FuncAnimation
from polygenerator import random_convex_polygon, random_polygon

from data_face_recognition import \
    gen_workflows as gen_workflows_face_recognition
from scheduler_gcn import schedule as schedule_gcn
from scheduler_heft import schedule as schedule_heft
from scheduler_rand import schedule as schedule_rand

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
    paths = dict(nx.all_pairs_dijkstra_path_length(graph, weight=lambda u, v, d: 1/d["bandwidth"]))
    for node1, node2 in nx.non_edges(graph):
        bandwidth = 1 / paths[node1][node2]
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
            scheduler: Optional[Callable[[nx.DiGraph, nx.Graph], Dict[Hashable, Hashable]]] = None) -> Generator[Tuple[nx.Graph, Optional[float]], None, None]:
        i = 0
        if get_workflow is not None:
            workflow = get_workflow()
        schedule: Dict[Hashable, Hashable] = None
        finish_times: Dict[Hashable, float] = {}
        sim_time = 0
        start_time = sim_time
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
            network = to_fully_connected(network)

            # Execute workflow assuming network is valid for next timestep time
            if get_workflow is not None and scheduler is not None:
                if schedule is None:
                    schedule = scheduler(workflow, network)
                finish_times, success = run_workflow(
                    workflow, network, schedule, finish_times, (sim_time-start_time)+self.timestep
                )
                if success:
                    yield network, max(finish_times.values())
                    workflow = get_workflow()
                    schedule = None
                    finish_times = {}
                    start_time = sim_time
                else:
                    yield network, None
            else:
                yield network, None
            
            i += 1

            # Stop if we have reached the end
            if rounds is not None and i >= rounds:
                break

def visual_sim(sim: Simulation, 
               rounds: Optional[int] = None,
               get_workflow: Optional[Callable[[], nx.DiGraph]] = None,
               scheduler: Optional[Callable[[nx.DiGraph, nx.Graph], Dict[Hashable, Hashable]]] = None) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.tight_layout()
    ax1.axis('equal')
    ax2.axis('equal')

    ax1.add_patch(patches.Polygon(sim.polygon.points, fill=False, edgecolor="black"))
    
    sim_run = sim.run(rounds, get_workflow, scheduler)
    init_network, _ = next(sim_run)
    xdata, ydata = list(zip(*[pos for _, pos in nx.get_node_attributes(init_network, "pos").items()]))
    ln, = ax1.plot(xdata, ydata, 'ro')

    # draw network on ax2
    network_pos = nx.circular_layout(init_network)
    nx.draw(init_network, network_pos, with_labels=True, ax=ax2)

    round = 0
    def update(frame):
        nonlocal round
        round += 1
        print(f"Round {round} - Sim Time {round*sim.timestep:.2f} seconds")
        network, _ = next(sim_run)
        xdata, ydata = list(zip(*[pos for _, pos in nx.get_node_attributes(network, "pos").items()]))
        ln.set_data(xdata, ydata)

        ax2.clear()
        nx.draw(network, network_pos, with_labels=True, ax=ax2)
        return ln,

    ani = FuncAnimation(fig, update, frames=100, interval=100)
    plt.show()

def main():
    num_nodes = 10
    timestep = 1
    perimeter = 100
    n_trips = 3 # number of trips around the polygon
    n_sims = 9 # number of simulations to run

    rows = []
    bw_rows = []
    bw_one_rows = []
    for i_sim in range(n_sims):

        # enough rounds for robots to make a round trip
        rounds = int(perimeter / timestep) * n_trips
        print('Simulating for {} rounds'.format(rounds))

        _vertices = random_polygon(num_nodes)
        # normalized polygon so agents traverse entire polygon in one unit of time
        perim = Polygon(_vertices).perimeter
        _vertices = [(x/perim*perimeter, y/perim*perimeter) for x, y in _vertices]
        polygon = Polygon(_vertices)

        print('Perimeter:', polygon.perimeter)

        sim = Simulation(
            polygon=polygon,
            agent_cpus=[1 for _ in range(num_nodes)],
            timestep=timestep,
            radius_threshold=perimeter * (1+0.01)/num_nodes,
            bandwidth=lambda x: 1/(1 + np.exp((x/perimeter*num_nodes-1)*5))
        )

        # visual_sim(sim, rounds=rounds)
        # return

        workflow = gen_workflows_face_recognition(1)[0]
        first_network, _ = next(sim.run())

        static_sched = schedule_heft(workflow, first_network)
        
        sim_run_gcn = sim.run(rounds=rounds, get_workflow=lambda: workflow, scheduler=schedule_gcn)
        sim_run_heft = sim.run(rounds=rounds, get_workflow=lambda: workflow, scheduler=schedule_heft)
        sim_run_static = sim.run(rounds=rounds, get_workflow=lambda: workflow, scheduler=lambda w, n: static_sched)
        sim_run_rand = sim.run(rounds=rounds, get_workflow=lambda: workflow, scheduler=schedule_rand)


        i = 0
        for frame_gcn, frame_heft, frame_static, frame_rand in zip(sim_run_gcn, sim_run_heft, sim_run_static, sim_run_rand):
            # print('Sim Time:', i*sim.timestep)
            sim_time = i * sim.timestep
            
            network, makespan_gcn = frame_gcn
            _, makespan_heft = frame_heft
            _, makespan_static = frame_static
            _, makespan_rand = frame_rand

            # Network Bandwidth Info
            avg_bandwidth = np.mean([network.edges[src, dst]["bandwidth"] for src, dst in network.edges])
            std_bandwidth = np.std([network.edges[src, dst]["bandwidth"] for src, dst in network.edges])
            bw_rows.append([i_sim, sim_time, avg_bandwidth, std_bandwidth])

            # Average bandwidth of node 0
            avg_bandwidth_one = np.mean([network.edges[src, dst]["bandwidth"] for src, dst in network.edges if src == 0])
            std_bandwidth_one = np.std([network.edges[src, dst]["bandwidth"] for src, dst in network.edges if src == 0])
            bw_one_rows.append([i_sim, sim_time, avg_bandwidth_one, std_bandwidth_one])

            if makespan_gcn is not None:
                print("GCN makespan:", makespan_gcn)
                rows.append([i_sim, sim_time, makespan_gcn, 'GCN'])

            if makespan_heft is not None:
                print("HEFT makespan:", makespan_heft)
                rows.append([i_sim, sim_time, makespan_heft, 'HEFT'])

            if makespan_static is not None:
                print("Static makespan:", makespan_static)
                rows.append([i_sim, sim_time, makespan_static, 'Static'])

            if makespan_rand is not None:
                print("Random makespan:", makespan_rand)
                rows.append([i_sim, sim_time, makespan_rand, 'Random'])
            
            i += 1

    savedir = thisdir.joinpath('data', 'results')
    df = pd.DataFrame(rows, columns=['Simulation', 'Time', 'Makespan', 'Scheduler'])
    df.to_csv(savedir.joinpath('makespan.csv'), index=False)


    # Plots
    plotsdir = thisdir.joinpath('data', 'plots')	
    plotsdir.mkdir(exist_ok=True, parents=True)

    # Makespan Plot
    fig = px.scatter(
        df, 
        x='Time', y='Makespan', 
        color='Scheduler', 
        facet_col='Simulation', 
        facet_col_wrap=int(np.sqrt(n_sims)),
        template='plotly_white',
        trendline='expanding',
        # trendline_options=dict(window=3)
    )
    fig.write_image(str(plotsdir.joinpath('makespan.png')))
    fig.show()

    # Bandwidth Plot
    df_bw = pd.DataFrame(bw_rows, columns=['Simulation', 'Time', 'Avg Bandwidth', 'Std Bandwidth'])
    df_bw.to_csv(savedir.joinpath('bandwidth.csv'), index=False)
    fig_bw = px.scatter(
        df_bw, 
        x='Time', y='Avg Bandwidth', 
        error_y='Std Bandwidth',
        facet_col='Simulation', 
        facet_col_wrap=int(np.sqrt(n_sims)),
        template='plotly_white'
    )
    fig_bw.write_image(str(plotsdir.joinpath('bandwidth.png')))
    fig_bw.show()

    # Single Node Bandwidth Plot
    df_bw_one = pd.DataFrame(bw_one_rows, columns=['Simulation', 'Time', 'Avg Bandwidth', 'Std Bandwidth'])
    df_bw_one.to_csv(savedir.joinpath('bandwidth_one.csv'), index=False)
    fig_bw_one = px.scatter(
        df_bw_one,
        x='Time', y='Avg Bandwidth',
        error_y='Std Bandwidth',
        facet_col='Simulation',
        facet_col_wrap=int(np.sqrt(n_sims)),
        template='plotly_white'
    )
    fig_bw_one.write_image(str(plotsdir.joinpath('bandwidth_one.png')))
    fig_bw_one.show()


if __name__ == '__main__':
    main()
