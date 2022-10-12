from calendar import different_locale
import itertools
import pathlib
import random
from typing import Dict, Generator, Hashable, Iterator

import dill as pickle
import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
from polygenerator import random_convex_polygon, random_polygon

from data_face_recognition import gen_workflows as gen_workflows_face_recognition
from data_epigenomics import gen_workflows as gen_workflows_epigenomics
from dataset import WorkflowsDataset
from scheduler_heft import heft, schedule as schedule_heft
from scheduler_rand import schedule as schedule_rand
from simulate import Polygon, Simulation, run_workflow, to_fully_connected

thisdir = pathlib.Path(__file__).parent.resolve()

def gen_networks(n_networks: int,
                 n_sim_rounds: int,
                 n_nodes: int,
                 min_vertices: int = 7, 
                 max_vertices: int = 10) -> Generator[nx.Graph, None, None]:
    """Generate n_networks * n_sim_rounds networks with n_nodes nodes
    
    Args:
        n_networks (int): Number of networks to generate
        n_sim_rounds (int): Number of simulation rounds to run
        n_nodes (int): Number of nodes in the network
        min_vertices (int, optional): Minimum number of vertices in the polygon. Defaults to 3.
        max_vertices (int, optional): Maximum number of vertices in the polygon. Defaults to 10.

    Returns:
        Generator[nx.Graph, None, None]: Generator of networks
    """
    for i in range(n_networks):
        n_vertices = random.randint(min_vertices, max_vertices)
        _vertices = random_polygon(n_vertices)
        # normalized polygon so agents traverse entire polygon in one unit of time
        perim = Polygon(_vertices).perimeter
        _vertices = [(x/perim, y/perim) for x, y in _vertices]
        polygon = Polygon(_vertices)

        agent_cpus = np.ones(n_nodes) # + np.random.normal(0, 0.01, n_nodes)
        # agent_cpus = np.random.triangular(1/2, 1, 3/2, n_nodes)
        # agent_cpus = np.zeros(n_nodes) + 1/100
        # agent_cpus[np.random.randint(0, n_nodes)] = 1

        perimeter = polygon.perimeter
        bandwidth_min = 0.2
        curvature = 5
        sim = Simulation(
            polygon=polygon,
            agent_cpus=agent_cpus,
            timestep=1/n_sim_rounds, # take networks spread evenly over one unit of time
            radius_threshold=(1+0.01)/n_nodes, # radius of communication
            bandwidth=lambda x: (1-bandwidth_min)/(1 + np.exp((2*x*n_nodes/perimeter-1)*curvature)) + bandwidth_min,
            deadzone=0.25
        )
        # visual_sim(sim)
        for network, _ in sim.run(rounds=n_sim_rounds):
            yield to_fully_connected(network)

def gen_samples(networks: Iterator[nx.Graph], workflows: Iterator[nx.DiGraph]) -> None:
    out_folder = thisdir.joinpath('data')
    out_folder.mkdir(exist_ok=True, parents=True)
    data = WorkflowsDataset(
        networks=networks, 
        workflows=workflows, 
        scheduler=schedule_heft
    )
    out_folder.joinpath('data.pkl').write_bytes(pickle.dumps(data))

def apply_schedule(workflow: nx.DiGraph, 
                   network: nx.Graph, 
                   schedule: Dict[Hashable, Hashable],
                   task_order: Iterator[Hashable]) -> None:
        nx.set_node_attributes(
            workflow, 
            {
                task_name: schedule[task_name] 
                for task_name, node in schedule.items()
            }, 
            'node'
        )
        for task_name in reversed(list(task_order)):
            network.nodes[schedule[task_name]].setdefault('tasks', []).append(task_name)

def validate_dataset(networks: Iterator[nx.Graph], workflows: Iterator[nx.DiGraph]) -> None:
    rows = []
    for i, (network, workflow) in enumerate(itertools.product(networks, workflows)):
        print(f'Sample {i+1}')
        heft_sched = schedule_heft(workflow, network)

        task_schedule, comp_schedule = heft(network, workflow) # this is LRU cached so it's fast
        task_order = sorted(list(task_schedule.keys()), key=lambda x: task_schedule[x].start)
        apply_schedule(workflow, network, heft_sched, task_order)
        heft_makespan = run_workflow(workflow.copy(), network.copy())

        rand_makespans = []
        for _ in range(10):
            rand_sched = schedule_rand(workflow, network)
            task_order = nx.topological_sort(workflow)
            apply_schedule(workflow, network, rand_sched, task_order)
            rand_makespans.append(run_workflow(workflow.copy(), network.copy()))
            
        rand_makespan = np.mean(rand_makespans)

        rows.append([i, heft_makespan, rand_makespan])

    df = pd.DataFrame(rows, columns=['sample', 'heft_makespan', 'rand_makespan'])
    print(df.to_string())
    fig = px.line(df, x='sample', y=['heft_makespan', 'rand_makespan'])
    fig.write_image(thisdir.joinpath('makespans.png'))
    fig.show()


def main():
    n_networks = 50
    n_sim_rounds = 10
    n_workflows = 10
    n_nodes = 10

    networks = gen_networks(n_networks, n_sim_rounds, n_nodes)
    # workflows = gen_workflows_face_recognition(n_workflows, n_copies=1)
    workflows = list(gen_workflows_epigenomics(n_workflows, n_copies=1))

    gen_samples(networks, workflows)
    # validate_dataset(networks, workflows)

if __name__ == '__main__':
    main()
