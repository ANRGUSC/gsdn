import itertools
import pathlib
from typing import Callable, Dict, Hashable, Iterable

import dgl
import networkx as nx
import pandas as pd
import torch
from dgl.data import DGLDataset

thisdir = pathlib.Path(__file__).parent.resolve()

def preprocess(workflow: nx.DiGraph,
               network: nx.Graph,
               task_cost: Callable[[Hashable, Hashable], float],
               schedule: Callable[[nx.DiGraph, nx.Graph], Dict[Hashable, Hashable]]):
    sched = schedule(workflow, network, task_cost)
    def comm_cost(src, dst, node_src, node_dst):
        if node_src == node_dst:
            return 0
        return workflow.edges[src, dst]['data'] / network.edges[node_src, node_dst]['bandwidth']

    df_tasks = pd.DataFrame(
        [
            [task, task_label, *[task_cost(task, node) for node in network.nodes]]
            for task, task_label in sched.items()
        ],
        columns=['task', 'task_label', *[f'Cost_{node}' for node in network.nodes]]
    )

    all_node_pairs = list(itertools.product(network.nodes, network.nodes))
    df_edges = pd.DataFrame(
        [
            [src, dst, *[comm_cost(src, dst, node_src, node_dst) for node_src, node_dst in all_node_pairs]]
            for src, dst in workflow.edges
        ],
        columns=['src', 'dst', *[f'Cost_{node_src}_{node_dst}' for node_src, node_dst in all_node_pairs]]
    )
    return df_tasks, df_edges

class WorkflowsDataset(DGLDataset):
    def __init__(self, 
                 workflows: Iterable[nx.DiGraph], 
                 networks: Iterable[nx.Graph],
                 scheduler: Callable[[nx.DiGraph, nx.Graph], Dict[Hashable, Hashable]]) -> None:
        self.networks = networks
        self.workflows = workflows
        self.scheduler = scheduler
        super().__init__(name='workflows')

    def process(self):
        edges_src, edges_dst = torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)
        node_labels = torch.tensor([], dtype=torch.long)
        node_features = torch.tensor([], dtype=torch.float32)
        edge_features = torch.tensor([], dtype=torch.float32)

        n_train = torch.tensor([], dtype=torch.long)
        n_val = torch.tensor([], dtype=torch.long)

        n_nodes = 0
        self.num_classes = 0
        for i, (network, workflow) in enumerate(itertools.product(self.networks, self.workflows)):
            print(f'Processing sample {i + 1}')
            tasks, edges = preprocess(
                workflow, network, 
                lambda task, node: workflow.nodes[task]['cost'] / network.nodes[node]['cpu'],
                self.scheduler
            )

            node_cost_cols = [col for col in tasks.columns if col.startswith('Cost_')]

            node_features = torch.cat((node_features, torch.from_numpy(tasks[node_cost_cols].to_numpy())), dim=0)
            node_labels = torch.cat((node_labels, torch.from_numpy(tasks['task_label'].astype('category').cat.codes.to_numpy())), dim=0)
                        
            edges_src = torch.cat((edges_src, torch.from_numpy(edges['src'].to_numpy() + n_nodes)), dim=0)
            edges_dst = torch.cat((edges_dst, torch.from_numpy(edges['dst'].to_numpy() + n_nodes)), dim=0)

            n_nodes += len(tasks)
            self.num_classes = network.number_of_nodes() # should be the same for all networks

            edge_cost_cols = [col for col in edges.columns if col.startswith('Cost_')]
            edge_features = torch.cat((edge_features, torch.from_numpy(edges[edge_cost_cols].to_numpy())), dim=0)

        # normalize features
        max_feature = max(node_features.max(), edge_features.max())
        min_feature = min(node_features.min(), edge_features.min())
        node_features = (node_features - min_feature) / (max_feature - min_feature)
        edge_features = (edge_features - min_feature) / (max_feature - min_feature)

        self.graph = dgl.graph((edges_src, edges_dst), num_nodes=n_nodes)
        self.graph.ndata['labels'] = node_labels
        self.graph.ndata['node_features'] = node_features
        self.graph.edata['edge_features'] = edge_features


        self.edge_dim = self.num_classes**2
        self.node_dim = self.num_classes

        # If your dataset is a node classification dataset, you will need to assign
        # masks indicating whether a node belongs to training, validation, and test set.
        n_train = int(n_nodes * 0.6)
        n_val = int(n_nodes * 0.2)
        train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        val_mask = torch.zeros(n_nodes, dtype=torch.bool)
        test_mask = torch.zeros(n_nodes, dtype=torch.bool)
        train_mask[:n_train] = True
        val_mask[n_train:n_train + n_val] = True
        test_mask[n_train + n_val:] = True
        self.graph.ndata['train_mask'] = train_mask
        self.graph.ndata['val_mask'] = val_mask
        self.graph.ndata['test_mask'] = test_mask

    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return 1
