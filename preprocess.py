import itertools
import pathlib
import random
from functools import partial
from typing import Callable, Dict, Generator, Hashable

import networkx as nx
import pandas as pd
import plotly.express as px
import numpy as np
import torch
from core.data.constants import (GRAPH, LABELS, N_CLASSES, TEST_MASK,
                                 TRAIN_MASK, VAL_MASK)
from core.data.utils import complete_path, save_pickle

from dataset import WorkflowsDataset
from heft import heft
from simulate import Polygon, Simulation, to_fully_connected, run_workflow, visual_sim
from data_wfchef import triangle_distribution, gen_workflows as gen_workflows_wfchef

from polygenerator import random_polygon, random_convex_polygon

thisdir = pathlib.Path(__file__).parent.resolve()

def schedule_heft(workflow: nx.DiGraph, 
                  network: nx.Graph,
                  task_cost: Callable[[Hashable, Hashable], float]) -> Dict[Hashable, Hashable]:
    """Schedule workflow on network using HEFT algorithm
    
    Args:
        workflow (nx.DiGraph): Workflow graph
        network (nx.Graph): Network graph
        task_cost (Callable[[Hashable, Hashable], float]): Function that returns the cost of
            running a task on a node. The function should take two arguments: the task and the
            node and return the cost of running the task on the node.

        Returns:
            Dict[Hashable, Hashable]: Mapping from task to node
    """
    task_schedule, comm_schedule, comp_schedule = heft(network, workflow, task_cost)
    return {task.name: node for node, tasks in comp_schedule.items() for task in tasks}

def schedule_random(workflow: nx.DiGraph, 
                    network: nx.Graph,
                    task_cost: Callable[[Hashable, Hashable], float]) -> Dict[Hashable, Hashable]:
    """Schedule workflow on network using random algorithm
    
    Args:
        workflow (nx.DiGraph): Workflow graph
        network (nx.Graph): Network graph
        task_cost (Callable[[Hashable, Hashable], float]): Function that returns the cost of
            running a task on a node. The function should take two arguments: the task and the
            node and return the cost of running the task on the node.

    Returns:
        Dict[Hashable, Hashable]: Mapping from task to node
    """
    return {task: random.choice(list(network.nodes)) for task in workflow.nodes} 

def gen_networks(n_networks: int,
                 n_sim_rounds: int,
                 n_nodes: int,
                 min_vertices: int = 3, 
                 max_vertices: int = 10,
                 bandwidth_scale_factor: float = 1.0) -> Generator[nx.Graph, None, None]:
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
        sim = Simulation(
            polygon=polygon,
            agent_cpus=[np.random.triangular(1/2, 1, 3/2) for _ in range(n_nodes)],
            timestep=1/n_sim_rounds, # take networks spread evenly over one unit of time
            radius_threshold=(1+0.01)/n_nodes, # radius of communication
            bandwidth=lambda x: bandwidth_scale_factor * 1/(1 + np.exp((x*n_nodes-1)*5)), # sigmoid function
        )
        # visual_sim(sim)
        for network in sim.run(rounds=n_sim_rounds):
            yield to_fully_connected(network)

def main():
    out_folder = thisdir.joinpath('data')
    out_folder.mkdir(exist_ok=True, parents=True)

    n_networks = 200
    n_sim_rounds = 1
    n_workflows = 3
    print(f'Generating {n_networks * n_sim_rounds * n_workflows} samples')
    data = WorkflowsDataset(
        networks=gen_networks(
            n_networks=n_networks, n_sim_rounds=n_sim_rounds, 
            n_nodes=10, min_vertices=3, max_vertices=10,
            bandwidth_scale_factor=5e5
        ),
        workflows=gen_workflows_wfchef(
            n_workflows=n_workflows, 
            recipe_name='epigenomics', 
            get_order=partial(triangle_distribution, 1.5)
        ),
        scheduler=schedule_heft
    )

    save_pickle(data.graph, complete_path(out_folder, GRAPH))
    save_pickle(data.num_classes, complete_path(out_folder, N_CLASSES))
    torch.save(data.graph.ndata['labels'], complete_path(out_folder, LABELS))
    torch.save(data.graph.ndata['train_mask'], complete_path(out_folder, TRAIN_MASK))
    torch.save(data.graph.ndata['val_mask'], complete_path(out_folder, TEST_MASK))
    torch.save(data.graph.ndata['test_mask'], complete_path(out_folder, VAL_MASK))


def validate_dataset() -> None:
    n_networks = 50
    n_sim_rounds = 1
    n_workflows = 1
    print(f'Generating {n_networks * n_sim_rounds * n_workflows} samples')
    networks = gen_networks(
        n_networks=n_networks, n_sim_rounds=n_sim_rounds, 
        n_nodes=10, min_vertices=3, max_vertices=10,
        bandwidth_scale_factor=1
    )
    workflows = gen_workflows_wfchef(
        n_workflows=n_workflows, 
        recipe_name='epigenomics', 
        get_order=partial(triangle_distribution, 1.5)
    )

    rows = []
    task_cost = lambda task, node: workflow.nodes[task]['cost'] / network.nodes[node]['cpu']
    for i, (network, workflow) in enumerate(itertools.product(networks, workflows)):
        # max_task_cost = max(workflow.nodes[task]['cost'] for task in workflow.nodes)
        # max_data_size = max(workflow.edges[task1, task2]['data'] for task1, task2 in workflow.edges)

        bandwidth_scale_factor = 5e5 # max_data_size / (max_task_cost*0.5)
        nx.set_edge_attributes(
            network, 
            {
                edge: bandwidth_scale_factor * network.edges[edge]['bandwidth'] 
                for edge in network.edges
            },
            'bandwidth'
        )

        print(f'[{i+1}/{n_networks * n_sim_rounds * n_workflows}]')
        heft_sched = schedule_heft(workflow, network, task_cost)
        # print(f'HEFT schedule: {heft_sched}')
        finish_times, _ = run_workflow(workflow, network, heft_sched)
        heft_makespan = max(finish_times.values())

        rand_makespan = np.mean(
            [
                max(
                    run_workflow(
                        workflow, network, schedule_random(workflow, network, task_cost)
                    )[0].values()
                )
                for _ in range(10)
            ]
        )

        rows.append([i, heft_makespan, rand_makespan, bandwidth_scale_factor])

    df = pd.DataFrame(rows, columns=['sample', 'heft_makespan', 'rand_makespan', 'bandwidth_scale_factor'])
    print('badwidth_scale_factor', df['bandwidth_scale_factor'].mean())
    print(df.to_string())
    fig = px.line(df, x='sample', y=['heft_makespan', 'rand_makespan'])
    fig.write_image(thisdir.joinpath('makespans.png'))
    fig.show()


if __name__ == '__main__':
    main()
    # validate_dataset()