from functools import partial
import os
import pathlib
import random
import shutil
import time
from typing import Callable, Dict, Generator, Hashable, Iterator, List, Optional, Tuple

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.style import available
import plotly.graph_objects as go
import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
from matplotlib.animation import FuncAnimation
from polygenerator import random_convex_polygon, random_polygon
import dill as pickle
from joblib import Parallel, delayed

from data_face_recognition import gen_workflows as gen_workflows_face_recognition
from data_epigenomics import gen_workflows as gen_workflows_epigenomics
from dataset import WorkflowsDataset
from scheduler_gcn import schedule as schedule_gcn
from scheduler_heft import heft, heft_rank_sort, schedule as schedule_heft
from scheduler_rand import schedule as schedule_rand

thisdir = pathlib.Path(__file__).parent.resolve()

def load_dataset_workflows() -> Tuple[List[nx.DiGraph], List[nx.DiGraph], List[nx.DiGraph], List[nx.DiGraph]]:
    data: WorkflowsDataset = pickle.loads(thisdir.joinpath('data', 'data.pkl').read_bytes())
    return list(data.workflows), list(data.train_workflows), list(data.val_workflows), list(data.test_workflows)

def load_dataset_networks() -> Tuple[List[nx.Graph], List[nx.Graph], List[nx.Graph], List[nx.Graph]]:
    data: WorkflowsDataset = pickle.loads(thisdir.joinpath('data', 'data.pkl').read_bytes())
    return list(data.networks), list(data.train_networks), list(data.val_networks), list(data.test_networks)

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
    # paths = dict(nx.all_pairs_dijkstra_path_length(graph, weight=lambda u, v, d: 1/d["bandwidth"]))
    # paths = nx.floyd_warshall_numpy(graph, weight=lambda u, v, d: 1/d["bandwidth"])
    # paths = nx.all_pairs_shortest_path_length(graph, weight=lambda u, v, d: 1/d["bandwidth"])
    for node1, node2 in nx.non_edges(graph):
        shortest_path_length = nx.shortest_path_length(graph, node1, node2, weight=lambda u, v, d: 1/d["bandwidth"])
        bandwidth = 1 / shortest_path_length
        graph.add_edge(node1, node2, bandwidth=bandwidth)
    return graph

def run_workflow(workflow: nx.DiGraph, 
                 network: nx.Graph, 
                 stop_time: float = np.inf) -> Optional[float]:
    """Get the makespan of a workflow on a network with a schedule.
    
    Args:
        workflow: The workflow to run. 
        network: The network to run the workflow on.
        stop_time: The time to stop the simulation at.
    Returns:
        The makespan of the workflow or None if the workflow could not be completed.
    """
    did_action = True
    while did_action:
        did_action = False
        # Continue existing executions
        for node in network.nodes:
            cur_task = network.nodes[node].get('task')
            if cur_task is None: # No task running on this node
                continue

            start_time = network.nodes[node]['start_time']
            runtime = workflow.nodes[cur_task]['cost'] / network.nodes[node]['cpu']
            if start_time + runtime > stop_time:
                continue # Can't finish task this timestep
        
            did_action = True
            workflow.nodes[cur_task]['end_time'] = start_time + runtime
            finish_time = start_time + runtime
            network.nodes[node]['start_time'] = finish_time
            network.nodes[node]['task'] = None

            # Queue new communication
            for child_task in workflow.successors(cur_task):
                next_node = workflow.nodes[child_task]['node']
                if next_node == node:
                    # Add current task to ready tasks on next node
                    network.nodes[node].setdefault('ready', {}).setdefault(child_task, {})[cur_task] = finish_time
                else:
                    # Add task to communication queue
                    network.nodes[next_node].setdefault('comm_progress', {})[(cur_task, child_task)] = (0, finish_time)
    
        # Continue communications
        for node in network.nodes:
            comm_progress = network.nodes[node].get('comm_progress', {})
            for (src_task, dst_task), (progress, last_time) in list(comm_progress.items()):
                if last_time >= stop_time:
                    continue # Skip if no time for more communication
                src_task_data: float = workflow.edges[src_task, dst_task]['data'] - progress
                src_node = workflow.nodes[src_task]['node']
                comm_rate: float = network.edges[src_node, node]['bandwidth']
                comm_time = src_task_data / comm_rate
                if last_time + comm_time > stop_time: # communication will continue into next time step
                    network.nodes[node]['comm_progress'][(src_task, dst_task)] = (progress + (stop_time-last_time) * comm_rate, stop_time)
                else:
                    # Add current task to ready tasks on next node
                    did_action = True
                    network.nodes[node].setdefault('ready', {}).setdefault(dst_task, {})[src_task] = last_time + comm_time
                    del network.nodes[node]['comm_progress'][(src_task, dst_task)]
        
        # Start new executions
        for node in network.nodes:
            if not network.nodes[node].get('tasks', []):
                continue # No more tasks scheduled on this node
            if network.nodes[node].get('task') is not None:
                continue # Task already running on this node
            task = network.nodes[node].get('tasks', [])[-1]
            ready: Dict[Hashable, Dict[Hashable, float]] = network.nodes[node].get('ready', {}).get(task, {})
            task_parents = set(workflow.predecessors(task))
            if task_parents.issubset(ready.keys()):
                # Start task
                did_action = True
                network.nodes[node].get('tasks', []).pop(-1)
                network.nodes[node]['task'] = task
                network.nodes[node]['start_time'] = max([
                    network.nodes[node].get('start_time', 0.0), 
                    *(ready[parent] for parent in task_parents)
                ])
                workflow.nodes[task]['start_time'] = network.nodes[node]['start_time']

    if all(workflow.nodes[task].get('end_time') is not None for task in workflow.nodes):
        return max(workflow.nodes[task]['end_time'] for task in workflow.nodes)
    return None 
        
class Simulation:
    def __init__(self,
                 polygon: Polygon, 
                 agent_cpus: int,
                 timestep: float,
                 radius_threshold: float,
                 bandwidth: Callable[[float], float],
                 deadzone: Optional[float] = None) -> None:
        self.polygon = polygon
        self.agent_cpus = agent_cpus
        self.num_agents = len(agent_cpus)
        self.timestep = timestep
        self.radius_threshold = radius_threshold
        self.bandwidth = bandwidth
        self.deadzone = deadzone

    def init_positions(self) -> List[Tuple[float, float]]:
        dists = np.arange(0, self.polygon.perimeter, self.polygon.perimeter / self.num_agents)
        return [self.polygon._get_point(d) for d in dists]

    def run(self, 
            rounds: Optional[int] = None,
            get_workflow: Optional[Callable[[], nx.DiGraph]] = None,
            scheduler: Optional[Callable[[nx.DiGraph, nx.Graph], Dict[Hashable, Hashable]]] = None,
            topological_sort: Callable[[nx.DiGraph], Iterator[Hashable]] = lambda _, wf: nx.topological_sort(wf),
            name: str = "Simulation") -> Generator[Tuple[nx.Graph, Optional[float]], None, None]:
        i = 0
        sim_time = 0.0
        workflow, network, start_time = None, None, 0.0
        schedule: Dict[Hashable, Hashable] = {}
        while True:
            sim_time = i * self.timestep
            # move Agents and get comm network
            dists = (np.arange(0, self.polygon.perimeter, self.polygon.perimeter / self.num_agents) + sim_time) % self.polygon.perimeter
            deadzone_set = {i for i, d in enumerate(dists) if self.deadzone is not None and d > (self.polygon.perimeter - self.deadzone * self.polygon.perimeter)}
            # print("Deadzone set:", deadzone_set)
            pos = [self.polygon._get_point(d) for d in dists]
            
            _network = nx.Graph()
            _network.add_nodes_from([
                (node, {"pos": p, "cpu": self.agent_cpus[node] if node not in deadzone_set else 1e-9}) 
                for node, p in zip(range(self.num_agents), pos)
            ])
            for src, dst in nx.geometric_edges(_network, radius=self.radius_threshold):
                d = np.sqrt((pos[src][0] - pos[dst][0]) ** 2 + (pos[src][1] - pos[dst][1]) ** 2)
                if src in deadzone_set or dst in deadzone_set:
                    _network.add_edge(src, dst, bandwidth=1e-9)
                else:
                    _network.add_edge(src, dst, bandwidth=self.bandwidth(d))
            _network = to_fully_connected(_network)

            if network is None: # new network - new schedule
                network = _network.copy()
                if get_workflow is not None and scheduler is not None:
                    start_time = sim_time
                    workflow = get_workflow().copy()
                    schedule = scheduler(workflow, _network)
                    nx.set_node_attributes(workflow, {task_name: schedule[task_name] for task_name, node in schedule.items()}, 'node')
                    for task_name in reversed(list(topological_sort(network, workflow))):
                        network.nodes[schedule[task_name]].setdefault('tasks', []).append(task_name)
            else: # same network - update bandwidths and positions
                for src, dst in _network.edges:
                    network.edges[src, dst]["bandwidth"] = _network.edges[src, dst]["bandwidth"]
                for node in network.nodes:
                    network.nodes[node]["pos"] = _network.nodes[node]["pos"]
                    network.nodes[node]["cpu"] = _network.nodes[node]["cpu"]

            if get_workflow is not None and scheduler is not None:
                makespan = run_workflow(workflow, network, (sim_time-start_time)+self.timestep)
                if makespan is not None:
                    yield network, makespan
                    workflow, network = None, None
                else:
                    yield network, None
            else:
                yield network, None
            
            if network is not None and name in ['GCN', 'HEFT', 'Random']:
                max_network_node_start_time = start_time + max(
                    network.nodes[node].get('start_time', 0.0) for node in network.nodes
                )
                # print(f"Network node start time: {max_network_node_start_time} <= {sim_time - 10}")
                if max_network_node_start_time <= sim_time - 10:
                    # print(f'{name} - dead schedule detected')
                    network = None

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

def run_sim(i_sim: int, num_nodes: int, perimeter: int, n_trips: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print(f"Running Simulation {i_sim}")
    timestep = perimeter / 100 # run each simulation for 100 rounds

    rounds = int(perimeter / timestep) * n_trips
    rows = []
    bw_rows = []
    bw_one_rows = []
    num_execs_rows = []

    _vertices = random_polygon(num_nodes)
    # normalized polygon so agents traverse entire polygon in one unit of time
    perim = Polygon(_vertices).perimeter
    _vertices = [(x/perim*perimeter, y/perim*perimeter) for x, y in _vertices]
    polygon = Polygon(_vertices)

    bandwidth_min = 0.2
    curvature = 5
    sim = Simulation(
        polygon=polygon,
        agent_cpus=np.ones(num_nodes),
        timestep=timestep,
        radius_threshold=perimeter * (1+0.01)/num_nodes,
        bandwidth=lambda x: (1-bandwidth_min)/(1 + np.exp((2*x*num_nodes/perimeter-1)*curvature)) + bandwidth_min,
        deadzone=0.25
    )

    # workflow = gen_workflows_face_recognition(1, n_copies=1)[0]
    workflow = gen_workflows_epigenomics(1, n_copies=1)[0]
    # *_, workflows = load_dataset_workflows()
    # workflow = random.choice(workflows)
    first_network, _ = next(sim.run())

    static_sched = schedule_heft(workflow, first_network)

    def heft_top_sort(network: nx.Graph, workflow: nx.DiGraph) -> List[Hashable]:
        task_schedule, comp_schedule = heft(network, workflow) # this is LRU cached so it's fast
        return sorted(list(task_schedule.keys()), key=lambda x: task_schedule[x].start)
    
    rows_schedule_times = []
    def schedule_timer(name: str, 
                       scheduler: Callable[[nx.DiGraph, nx.Graph], Dict[Hashable, Hashable]],
                       workflow: nx.DiGraph,
                       network: nx.Graph) -> float:
        start = time.time()
        schedule = scheduler(workflow, network)
        rows_schedule_times.append([name, time.time()-start])
        return schedule

    sim_run_gcn = sim.run(
        rounds=rounds, get_workflow=lambda: workflow, 
        scheduler=partial(schedule_timer, 'GCN', schedule_gcn),
        topological_sort=heft_top_sort,
        name='GCN'
    )
    sim_run_heft = sim.run(
        rounds=rounds, get_workflow=lambda: workflow, 
        scheduler=partial(schedule_timer, 'HEFT', schedule_heft), 
        topological_sort=heft_top_sort,
        name='HEFT'
    )
    sim_run_static = sim.run(
        rounds=rounds, get_workflow=lambda: workflow, 
        scheduler=partial(schedule_timer, 'Static', lambda w, n: static_sched),
        name='Static'
    )
    sim_run_rand = sim.run(
        rounds=rounds, get_workflow=lambda: workflow, 
        scheduler=partial(schedule_timer, 'Random', schedule_rand),
        name='Random'
    )

    i = 0
    gcn_execs, heft_execs, static_execs, rand_execs = 0, 0, 0, 0
    for frame_gcn, frame_heft, frame_static, frame_rand in zip(sim_run_gcn, sim_run_heft, sim_run_static, sim_run_rand):
        sim_time = i * sim.timestep
        
        network, makespan_gcn = frame_gcn
        _, makespan_heft = frame_heft
        _, makespan_static = frame_static
        _, makespan_rand = frame_rand

        # Network Bandwidth Info
        avg_bandwidth = np.mean([network.edges[src, dst]["bandwidth"] for src, dst in network.edges])
        std_bandwidth = np.std([network.edges[src, dst]["bandwidth"] for src, dst in network.edges])
        bw_rows.append([i_sim, sim_time, avg_bandwidth, std_bandwidth])

        # Number of executions

        # Average bandwidth of node 0
        # avg_bandwidth_one = np.mean([network.edges[0, dst]["bandwidth"] for dst in network.nodes])
        avg_bandwidth_one = np.mean([network.edges[src, dst]["bandwidth"] for src, dst in network.edges if src == 0])
        std_bandwidth_one = np.std([network.edges[src, dst]["bandwidth"] for src, dst in network.edges if src == 0])
        bw_one_rows.append([i_sim, sim_time, avg_bandwidth_one, std_bandwidth_one])

        if makespan_gcn is not None:
            rows.append([i_sim, sim_time, makespan_gcn, 'GCN'])
            gcn_execs += 1

        if makespan_heft is not None:
            rows.append([i_sim, sim_time, makespan_heft, 'HEFT'])
            heft_execs += 1

        if makespan_static is not None:
            rows.append([i_sim, sim_time, makespan_static, 'Static'])
            static_execs += 1

        if makespan_rand is not None:
            rows.append([i_sim, sim_time, makespan_rand, 'Random'])
            rand_execs += 1
        
        i += 1

    df = pd.DataFrame(rows, columns=['Simulation', 'Time', 'Makespan', 'Scheduler'])
    df_bw = pd.DataFrame(bw_rows, columns=['Simulation', 'Time', 'Avg Bandwidth', 'Std Bandwidth'])
    df_bw_one = pd.DataFrame(bw_one_rows, columns=['Simulation', 'Time', 'Avg Bandwidth', 'Std Bandwidth'])
    df_schedule_times = pd.DataFrame(rows_schedule_times, columns=['Scheduler', 'Time'])
    df_num_execs = pd.DataFrame(
        [[i_sim, gcn_execs, heft_execs, static_execs, rand_execs]],
        columns=['Simulation', 'GCN', 'HEFT', 'Static', 'Random']
    )

    return df, df_bw, df_bw_one, df_schedule_times, df_num_execs

def hex_to_rgba_str(hex: str, alpha: float) -> str:
    hex = hex.lstrip('#')
    hlen = len(hex)
    return f'rgba({", ".join(str(int(hex[i:i+hlen//3], 16)) for i in range(0, hlen, hlen//3))}, {alpha})'

def main():
    num_nodes = 10
    perimeter = 500
    n_trips = 3 # number of trips around the polygon
    n_sims = 50 # number of simulations to run

    results = Parallel(n_jobs=-2)(
        delayed(run_sim)(i_sim+1, num_nodes, perimeter, n_trips) 
        for i_sim in range(n_sims)
    )
    df = pd.concat([r[0] for r in results])
    df_bw = pd.concat([r[1] for r in results])
    df_bw_one = pd.concat([r[2] for r in results])
    df_schedule_times = pd.concat([r[3] for r in results])
    df_num_execs = pd.concat([r[4] for r in results])

    # Save Data
    savedir = thisdir.joinpath('data', 'results')
    savedir.mkdir(parents=True, exist_ok=True)
    df.to_csv(savedir.joinpath('makespan.csv'), index=False)
    df_bw.to_csv(savedir.joinpath('bandwidth.csv'), index=False)
    df_bw_one.to_csv(savedir.joinpath('bandwidth_one.csv'), index=False)
    df_schedule_times.to_csv(savedir.joinpath('schedule_times.csv'), index=False)
    df_num_execs.to_csv(savedir.joinpath('num_execs.csv'), index=False)

    # Average Makespans
    df_mean = df.drop(columns=['Time']).groupby(['Scheduler', 'Simulation']).mean().reset_index()
    df_mean = df_mean.set_index('Simulation').sort_index()
    df_mean = df_mean.pivot_table(values='Makespan', index='Simulation', columns='Scheduler')
    df_mean = df_mean.div(df_mean['HEFT'], axis=0)
    df_mean.to_csv(savedir.joinpath('makespan_mean.csv'))
    
if __name__ == '__main__':
    main()
