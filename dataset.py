import enum
import itertools
import os
import pathlib
import time
from typing import Callable, Dict, Hashable, Iterable

import dgl
from joblib import Parallel, delayed
import networkx as nx
import numpy as np
import pandas as pd
import torch
from dgl.data import DGLDataset

thisdir = pathlib.Path(__file__).parent.resolve()

def preprocess(workflow: nx.DiGraph,
               network: nx.Graph,
               schedule: Callable[[nx.DiGraph, nx.Graph], Dict[Hashable, Hashable]]):
    sched = schedule(workflow, network)
    def comm_cost(src, dst, node_src, node_dst):
        if node_src == node_dst:
            return 0
        return workflow.edges[src, dst]['data'] / network.edges[node_src, node_dst]['bandwidth']

    sorted_nodes = sorted(list(network.nodes))
    sorted_tasks = sorted(list(workflow.nodes))
    df_tasks = pd.DataFrame(
        [
            [task, sched[task], *[
                workflow.nodes[task]['cost'] / network.nodes[node]['cpu']
                for node in sorted_nodes
            ]]
            for task in sorted_tasks
        ],
        columns=['task', 'task_label', *[f'Cost_{node}' for node in network.nodes]]
    )

    all_node_pairs = list(itertools.product(sorted_nodes, sorted_nodes))
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
        self.networks = list(networks)
        self.workflows = list(workflows)
        self.scheduler = scheduler
        super().__init__(name='workflows')

    def process(self):
        n_networks_train = int(len(self.networks) * 0.6)
        n_networks_val = int(len(self.networks) * 0.2)
        self.train_networks = self.networks[:n_networks_train]
        self.val_networks = self.networks[n_networks_train:n_networks_train + n_networks_val]
        self.test_networks = self.networks[n_networks_train + n_networks_val:]

        n_workflows_train = int(len(self.workflows) * 0.6)
        n_workflows_val = int(len(self.workflows) * 0.2)
        self.train_workflows = self.workflows[:n_workflows_train]
        self.val_workflows = self.workflows[n_workflows_train:n_workflows_train + n_workflows_val]
        self.test_workflows = self.workflows[n_workflows_train + n_workflows_val:]

        print(f'Generating {len(self.train_networks)} training networks, {len(self.val_networks)} validation networks, and {len(self.test_networks)} test networks')
        print(f'Generating {len(self.train_workflows)} training workflows, {len(self.val_workflows)} validation workflows, and {len(self.test_workflows)} test workflows')

        all_pairs = list(itertools.chain(
            itertools.product(self.train_workflows, self.train_networks),
            itertools.product(self.val_workflows, self.val_networks),
            itertools.product(self.test_workflows, self.test_networks)
        ))
        id_offsets = np.cumsum([0, *(workflow.number_of_nodes() for workflow, _ in all_pairs)])
        self.num_classes = self.networks[0].number_of_nodes()
        def process_pair(i_pair: int, workflow: nx.DiGraph, network: nx.Graph) -> None:
            nonlocal id_offsets
            print(f'Processing pair {i_pair + 1}/{len(all_pairs)}')
            tasks, edges = preprocess(workflow, network, self.scheduler)

            node_cost_cols = [col for col in tasks.columns if col.startswith('Cost_')]

            node_features = torch.from_numpy(tasks[node_cost_cols].to_numpy())
            node_labels = torch.from_numpy(tasks['task_label'].to_numpy())
                        
            edges_src = torch.from_numpy(edges['src'].to_numpy() + id_offsets[i_pair])
            edges_dst = torch.from_numpy(edges['dst'].to_numpy() + id_offsets[i_pair])

            edge_cost_cols = [col for col in edges.columns if col.startswith('Cost_')]
            edge_features = torch.from_numpy(edges[edge_cost_cols].to_numpy())

            is_train = i_pair < len(self.train_workflows) * len(self.train_networks)
            is_val = i_pair < len(self.train_workflows) * len(self.train_networks) + len(self.val_workflows) * len(self.val_networks) and not is_train
            is_test = (
                i_pair < len(self.train_workflows) * len(self.train_networks) + 
                len(self.val_workflows) * len(self.val_networks) + 
                len(self.test_workflows) * len(self.test_networks)
            ) and not is_train and not is_val
            train_mask = torch.tensor([is_train]*workflow.number_of_nodes())
            val_mask = torch.tensor([is_val]*workflow.number_of_nodes())
            test_mask = torch.tensor([is_test]*workflow.number_of_nodes())

            return node_features, node_labels, edges_src, edges_dst, edge_features, train_mask, val_mask, test_mask

        print('Starting Preprocessing')
        n_jobs = 1 # for some reason 1 job is fastest
        batch_size = 'auto' if n_jobs in {0, 1} else int(len(all_pairs)/(n_jobs if n_jobs > 0 else os.cpu_count() + n_jobs + 1))
        results = Parallel(n_jobs=n_jobs, batch_size=batch_size)(
            delayed(process_pair)(i_pair, workflow, network) 
            for i_pair, (workflow, network) in enumerate(all_pairs)
        )

        node_features = torch.cat([result[0] for result in results]).float()
        node_labels = torch.cat([result[1] for result in results]).long()
        edges_src = torch.cat([result[2] for result in results]).long()
        edges_dst = torch.cat([result[3] for result in results]).long()
        edge_features = torch.cat([result[4] for result in results]).float()
        train_mask = torch.cat([result[5] for result in results]).bool()
        val_mask = torch.cat([result[6] for result in results]).bool()
        test_mask = torch.cat([result[7] for result in results]).bool()

        print(node_labels)

        # normalize features
        max_feature = max(node_features.max(), edge_features.max())
        min_feature = min(node_features.min(), edge_features.min())
        node_features = (node_features - min_feature) / (max_feature - min_feature)
        edge_features = (edge_features - min_feature) / (max_feature - min_feature)

        self.graph = dgl.graph((edges_src, edges_dst), num_nodes=id_offsets[-1])
        self.graph.ndata['labels'] = node_labels
        self.graph.ndata['node_features'] = node_features
        self.graph.edata['edge_features'] = edge_features

        self.edge_dim = self.num_classes**2
        self.node_dim = self.num_classes

        self.graph.ndata['train_mask'] = train_mask
        self.graph.ndata['val_mask'] = val_mask
        self.graph.ndata['test_mask'] = test_mask

    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return 1
