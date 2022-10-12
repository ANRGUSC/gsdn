import itertools
import pathlib
from functools import lru_cache
from typing import List

import dgl
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import plotly.express as px
import torch
import networkx as nx

from data_face_recognition import gen_workflows as gen_workflows_face_recognition
from data_epigenomics import gen_workflows as gen_workflows_epigenomics
from preprocess import gen_networks
from scheduler_gcn import schedule as schedule_gcn
from scheduler_heft import schedule as schedule_heft, heft
from scheduler_rand import schedule as schedule_rand
from simulate import run_workflow, load_dataset_workflows, load_dataset_networks

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

def evaluate_sample(i_sample: int, _workflow: nx.DiGraph, _network: nx.Graph) -> List:
    task_schedule, comp_schedule = heft(_network, _workflow)
    heft_sched = {task.name: task.node for task in task_schedule.values()}
    # print(f'HEFT: {heft_sched}')

    network = _network.copy()
    workflow = _workflow.copy()
    nx.set_node_attributes(workflow, {task_name: node for task_name, node in heft_sched.items()}, 'node')
    for task_name in reversed(sorted(list(task_schedule.keys()), key=lambda x: task_schedule[x].start)):
        network.nodes[heft_sched[task_name]].setdefault('tasks', []).append(task_name)
    heft_makespan = run_workflow(workflow, network)

    # Schedule with GCNScheduler
    network = _network.copy()
    workflow = _workflow.copy()
    gcn_sched = schedule_gcn(workflow, network)
    # print(f'GCN: {gcn_sched}')

    nx.set_node_attributes(workflow, {task_name: node for task_name, node in gcn_sched.items()}, 'node')
    for task_name in reversed(list(nx.topological_sort(workflow))):
        network.nodes[gcn_sched[task_name]].setdefault('tasks', []).append(task_name)

    # print("GCN SCHED", dict(sorted(gcn_sched.items())))
    gcn_makespan = run_workflow(workflow, network)

    # Get GCNScheduler Accuracy
    accuracy = sum(
        1 if heft_sched[task] == gcn_sched[task] else 0 
        for task in gcn_sched.keys()
    ) / len(gcn_sched)

    # Schedule with Random Scheduler (for 10 trials)
    rand_makespans = []
    for _ in range(10):
        rand_sched = schedule_rand(workflow, network)
        network = _network.copy()
        workflow = _workflow.copy()
        nx.set_node_attributes(workflow, {task_name: node for task_name, node in rand_sched.items()}, 'node')
        for task_name in reversed(list(nx.topological_sort(workflow))):
            network.nodes[rand_sched[task_name]].setdefault('tasks', []).append(task_name)

        rand_makespans.append(run_workflow(workflow, network))

    rand_makespan = np.mean(rand_makespans) 
    return [i_sample, heft_makespan, rand_makespan, gcn_makespan, accuracy]

def evaluate_model() -> None:
    n_networks = 10
    n_sim_rounds = 10
    n_workflows = 10
    n_nodes = 10

    networks = gen_networks(n_networks, n_sim_rounds, n_nodes=n_nodes)
    workflows = gen_workflows_epigenomics(n_workflows, n_copies=1)
    rows = Parallel(n_jobs=-2)(
        delayed(evaluate_sample)(i_sample, workflow, network) 
        for i_sample, (workflow, network) in enumerate(itertools.product(workflows, networks))
    )
    df = pd.DataFrame(rows, columns=['sample', 'HEFT', 'Random', 'GCNScheduler', 'accuracy'])
    print(df.to_string())
    print("Average accuracy:", df['accuracy'].mean())
    fig = px.line(
        df, x='sample', y=['HEFT', 'Random', 'GCNScheduler'],
        labels={'sample': 'Sample', 'value': 'Makespan', 'variable': 'Scheduler'}
    )

    savedir = thisdir.joinpath('data', 'results')
    savedir.mkdir(parents=True, exist_ok=True)
    plotsdir_html = thisdir.joinpath('data', 'plots', 'html')
    plotsdir_html.mkdir(parents=True, exist_ok=True)
    plotsdir_png = thisdir.joinpath('data', 'plots', 'images')
    plotsdir_png.mkdir(parents=True, exist_ok=True)

    df.to_csv(savedir.joinpath('results.csv'), index=False)
    fig.write_image(plotsdir_png.joinpath('no_sim_makespans.png'))
    fig.write_html(plotsdir_html.joinpath('no_sim_makespans.html'))




if __name__ == '__main__':
    evaluate_model()
