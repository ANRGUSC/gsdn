#!/usr/bin/env python3
"""
Run model script.
"""
import itertools
import pathlib
from functools import lru_cache

import dgl
import numpy as np
import pandas as pd
import plotly.express as px
import torch

from data_face_recognition import \
    gen_workflows as gen_workflows_face_recognition
from preprocess import gen_networks
from scheduler_gcn import schedule as schedule_gcn
from scheduler_heft import schedule as schedule_heft
from scheduler_rand import schedule as schedule_rand
from simulate import run_workflow

thisdir = pathlib.Path(__file__).parent.resolve()

# Change These Parameters

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
    return torch.load(thisdir.joinpath('data', 'model.pt'))

def evaluate_model() -> None:
    n_networks = 50
    n_sim_rounds = 1
    n_workflows = 1
    print(f'Generating {n_networks * n_sim_rounds * n_workflows} samples')
    networks = gen_networks(
        n_networks=n_networks, n_sim_rounds=n_sim_rounds, 
        n_nodes=10, min_vertices=3, max_vertices=10
    )
    workflows = gen_workflows_face_recognition(n_workflows)
    # workflows = gen_workflows_wfchef(
    #     n_workflows=n_workflows, 
    #     recipe_name='epigenomics', 
    #     get_order=partial(triangle_distribution, 1.5)
    # )

    rows = []
    for i, (network, workflow) in enumerate(itertools.product(networks, workflows)):
        # Schedule with HEFT
        heft_sched = schedule_heft(workflow, network)
        print("HEFT SCHED", dict(sorted(heft_sched.items())))
        finish_times, _ = run_workflow(workflow, network, heft_sched)
        heft_makespan = max(finish_times.values())

        # Schedule with GCNScheduler
        gcn_sched = schedule_gcn(workflow, network)
        print("GCN SCHED", dict(sorted(gcn_sched.items())))
        finish_times, _ = run_workflow(workflow, network, gcn_sched)
        gcn_makespan = max(finish_times.values())

        # Get GCNScheduler Accuracy
        accuracy = sum(
            1 if heft_sched[task] == gcn_sched[task] else 0 
            for task in gcn_sched.keys()
        ) / len(workflow.nodes)

        # Schedule with Random Scheduler (for 10 trials)
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

        rows.append([i, heft_makespan, rand_makespan, gcn_makespan, accuracy])

    df = pd.DataFrame(rows, columns=['sample', 'HEFT', 'Random', 'GCNScheduler', 'accuracy'])
    print(df.to_string())
    print("Average accuracy:", df['accuracy'].mean())
    fig = px.line(
        df, x='sample', y=['HEFT', 'Random', 'GCNScheduler'],
        labels={'sample': 'Sample', 'value': 'Makespan', 'variable': 'Scheduler'}
    )
    fig.write_image(thisdir.joinpath('makespans.png'))
    fig.show()



if __name__ == '__main__':
    evaluate_model()
