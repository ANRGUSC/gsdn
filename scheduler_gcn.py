import pathlib
from functools import lru_cache
from typing import Dict, Hashable

import dgl
import networkx as nx
import pandas as pd
import torch

from dataset import preprocess
from model import Model
from train import MODEL

thisdir = pathlib.Path(__file__).parent.resolve()

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

def schedule(workflow: nx.DiGraph, 
             network: nx.Graph) -> Dict[Hashable, Hashable]:
    """Schedule workflow on network using GCNScheduler
    
    Args:
        workflow (nx.DiGraph): Workflow graph
        network (nx.Graph): Network graph

        Returns:
            Dict[Hashable, Hashable]: Mapping from task to node
    """        # Preprocess data and schedule with HEFT
    dummy_schedule = lambda *_: {task: 0 for task in workflow.nodes}
    tasks, edges = preprocess(workflow, network, schedule=dummy_schedule)
    
    task_ids = {task: i for i, task in enumerate(tasks['task'])}
    r_task_ids = {i: task for task, i in task_ids.items()}

    graph = format_data(tasks, edges)
    config_params = {
        **MODEL,
        "edge_dim": network.number_of_nodes()**2,
        "node_dim": network.number_of_nodes()
    }

    model = Model(
        g=graph,
        config_params=config_params,
        n_classes=network.number_of_nodes(),
        is_cuda=False
    )
    model.load_state_dict(load_weights())
    with torch.no_grad():
        logits = model(None)
        _, indices = torch.max(logits, dim=1)

    return {r_task_ids[i]: node for i, node in enumerate(indices.tolist())}
