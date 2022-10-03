import itertools
import pathlib
import random
from typing import Generator, Iterator

import dill as pickle
import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
from polygenerator import random_convex_polygon, random_polygon

from data_face_recognition import gen_workflows as gen_workflows_face_recognition
from dataset import WorkflowsDataset
from scheduler_heft import schedule as schedule_heft
from scheduler_rand import schedule as schedule_rand
from simulate import Polygon, Simulation, run_workflow, to_fully_connected

thisdir = pathlib.Path(__file__).parent.resolve()

def gen_networks(n_networks: int,
                 n_sim_rounds: int,
                 n_nodes: int,
                 min_vertices: int = 3, 
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
        sim = Simulation(
            polygon=polygon,
            agent_cpus=np.ones(n_nodes),# [np.random.triangular(1/2, 1, 3/2) for _ in range(n_nodes)],
            timestep=1/n_sim_rounds, # take networks spread evenly over one unit of time
            radius_threshold=(1+0.01)/n_nodes, # radius of communication
            bandwidth=lambda x: 1/(1 + np.exp((x*n_nodes-1)*5)), # sigmoid function
        )
        # visual_sim(sim)
        for network, _ in sim.run(rounds=n_sim_rounds):
            print('Generated network')
            yield to_fully_connected(network)

def gen_samples(networks: Iterator[nx.Graph], workflows: Iterator[nx.DiGraph]) -> None:
    out_folder = thisdir.joinpath('data')
    out_folder.mkdir(exist_ok=True, parents=True)
    data = WorkflowsDataset(networks=networks, workflows=workflows, scheduler=schedule_heft)
    out_folder.joinpath('data.pkl').write_bytes(pickle.dumps(data))

def validate_dataset(networks: Iterator[nx.Graph], workflows: Iterator[nx.DiGraph]) -> None:
    rows = []
    for i, (network, workflow) in enumerate(itertools.product(networks, workflows)):
        print(f'Sample {i+1}')
        heft_sched = schedule_heft(workflow, network)
        # print(f'HEFT schedule: {heft_sched}')
        finish_times, _ = run_workflow(workflow, network, heft_sched)
        heft_makespan = max(finish_times.values())

        rand_makespan = np.mean(
            [
                max(
                    run_workflow(
                        workflow, network, schedule_rand(workflow, network)
                    )[0].values()
                )
                for _ in range(10)
            ]
        )

        rows.append([i, heft_makespan, rand_makespan])

    df = pd.DataFrame(rows, columns=['sample', 'heft_makespan', 'rand_makespan'])
    print(df.to_string())
    fig = px.line(df, x='sample', y=['heft_makespan', 'rand_makespan'])
    fig.write_image(thisdir.joinpath('makespans.png'))
    fig.show()


def main():
    n_networks = 50
    n_sim_rounds = 1
    n_workflows = 1
    n_nodes = 100

    print(f'Generating {n_networks * n_sim_rounds * n_workflows} samples')
    networks = gen_networks(
        n_networks=n_networks, n_sim_rounds=n_sim_rounds, 
        n_nodes=n_nodes, min_vertices=3, max_vertices=10
    )
    workflows = gen_workflows_face_recognition(n_workflows)

    # gen_samples(networks, workflows)
    validate_dataset(networks, workflows)

if __name__ == '__main__':
    main()
