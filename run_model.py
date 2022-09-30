#!/usr/bin/env python3
"""
Run model script.
"""
from functools import lru_cache, partial
import itertools
import pathlib
import pandas as pd
import numpy as np
import plotly.express as px

import torch

from core.models.constants import NODE_CLASSIFICATION
from core.models.model import Model
import networkx as nx
from dataset import preprocess
import dgl
from simulate import run_workflow
from train import MODEL


from preprocess import gen_networks, schedule_heft, schedule_random
from data_wfchef import gen_workflows, triangle_distribution 

from data_wfchef import gen_workflows as gen_workflows_wfchef

thisdir = pathlib.Path(__file__).parent.resolve()

# Change These Parameters
GPU = 0

def format_data(tasks: pd.DataFrame, edges: pd.DataFrame) -> dgl.DGLGraph:
    node_cost_cols = [col for col in tasks.columns if col.startswith('Cost_')]

    node_features = torch.from_numpy(tasks[node_cost_cols].to_numpy())
    edges_src = torch.from_numpy(edges['src'].to_numpy())
    edges_dst = torch.from_numpy(edges['dst'].to_numpy())
    
    edge_cost_cols = [col for col in edges.columns if col.startswith('Cost_')]
    edge_features = torch.from_numpy(edges[edge_cost_cols].to_numpy())

    graph = dgl.graph((edges_src, edges_dst), num_nodes=len(tasks))
    graph.ndata['node_features'] = node_features
    graph.edata['edge_features'] = edge_features

    node_labels = torch.from_numpy(tasks['task_label'].to_numpy())
    graph.ndata['labels'] = node_labels
    
    return graph

@lru_cache(maxsize=1)
def load_weights():
    return torch.load(thisdir.joinpath('model.pt'))

def evaluate_model() -> None:
    if GPU < 0:
        cuda = False
    else:
        cuda = True
        torch.cuda.set_device(GPU)

    n_networks = 10
    n_sim_rounds = 1
    n_workflows = 2
    print(f'Generating {n_networks * n_sim_rounds * n_workflows} samples')
    networks = gen_networks(
        n_networks=n_networks, n_sim_rounds=n_sim_rounds, 
        n_nodes=10, min_vertices=3, max_vertices=10,
        bandwidth_scale_factor=5e5
    )
    workflows = gen_workflows_wfchef(
        n_workflows=n_workflows, 
        recipe_name='epigenomics', 
        get_order=partial(triangle_distribution, 1.5)
    )

    rows = []
    task_cost = lambda task, node: workflow.nodes[task]['cost'] / network.nodes[node]['cpu']
    for i, (network, workflow) in enumerate(itertools.product(networks, workflows)):
        # Preprocess data and schedule with HEFT
        tasks, edges = preprocess(
            workflow, network, 
            lambda task, node: workflow.nodes[task]['cost'] / network.nodes[node]['cpu'],
            schedule=schedule_heft # scheduler doesn't matter
        )
        # tasks = tasks.set_index('task').sort_index()
        task_ids = {task: i for i, task in enumerate(tasks['task'])}
        r_task_ids = {i: task for task, i in task_ids.items()}
        heft_sched = {r_task_ids[i]: task_label for i, task_label in enumerate(tasks['task_label'])}
        # heft_sched = {i: task_label for i, task_label in enumerate(tasks['task_label'])}
        finish_times, _ = run_workflow(workflow, network, heft_sched)
        heft_makespan = max(finish_times.values())

        # Schedule with GCNScheduler
        graph = format_data(tasks, edges)
        config_params = {
            **MODEL,
            "edge_dim": 1,
            "node_dim": 1
        }

        config_params = {
            **MODEL,
            "edge_dim": network.number_of_nodes()**2,
            "node_dim": network.number_of_nodes()
        }
        model = Model(
            g=graph,
            config_params=config_params,
            n_classes=network.number_of_nodes(),
            is_cuda=cuda,
            mode=NODE_CLASSIFICATION
        )
        model.load_state_dict(load_weights())
        with torch.no_grad():
            logits = model(None)
            _, indices = torch.max(logits, dim=1)

        gcn_sched = {r_task_ids[i]: node for i, node in enumerate(indices.tolist())}
        finish_times, _ = run_workflow(workflow, network, gcn_sched)
        gcn_makespan = max(finish_times.values())

        # Get GCNScheduler Accuracy
        accuracy = sum(
            1 if heft_sched[task] == gcn_sched[task] else 0 
            for task in gcn_sched.keys()
        ) / len(workflow.nodes)

        # Get Random Scheduler Makespan (for 10 trials)
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

        rows.append([i, heft_makespan, rand_makespan, gcn_makespan, accuracy])

    df = pd.DataFrame(rows, columns=['sample', 'heft_makespan', 'rand_makespan', 'gcn_makespan', 'accuracy'])
    print(df.to_string())
    print("Average accuracy:", df['accuracy'].mean())
    fig = px.line(df, x='sample', y=['heft_makespan', 'rand_makespan', 'gcn_makespan'])
    fig.write_image(thisdir.joinpath('makespans.png'))
    fig.show()



if __name__ == '__main__':
    evaluate_model()
