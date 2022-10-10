import enum
import itertools
import pathlib
import time
from typing import Callable, Dict, Hashable, Iterable

import dgl
import networkx as nx
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

    df_tasks = pd.DataFrame(
        [
            [task, sched[task], *[
                workflow.nodes[task]['cost'] / network.nodes[node]['cpu']
                for node in sorted(list(network.nodes))
            ]]
            for task in sorted(list(workflow.nodes))
        ],
        columns=['task', 'task_label', *[f'Cost_{node}' for node in network.nodes]]
    )

    all_node_pairs = list(itertools.product(sorted(list(network.nodes)), sorted(list(network.nodes))))
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
        edges_src, edges_dst = torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)
        node_labels = torch.tensor([], dtype=torch.long)
        node_features = torch.tensor([], dtype=torch.float32)
        edge_features = torch.tensor([], dtype=torch.float32)
        train_mask = torch.BoolTensor()
        val_mask = torch.BoolTensor()
        test_mask = torch.BoolTensor()

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

        # self.train_workflows = self.workflows
        # self.val_workflows = self.workflows
        # self.test_workflows = self.workflows

        print(f'Processing {len(self.train_networks)} training networks, {len(self.val_networks)} validation networks, and {len(self.test_networks)} test networks')
        print(f'Processing {len(self.train_workflows)} training workflows, {len(self.val_workflows)} validation workflows, and {len(self.test_workflows)} test workflows')

        n_tasks = 0
        self.num_classes = 0
        def process_pair(workflow: nx.DiGraph, network: nx.Graph) -> None:
            nonlocal n_tasks, node_labels, node_features, edge_features, edges_src, edges_dst, train_mask, val_mask, test_mask
            tasks, edges = preprocess(workflow, network, self.scheduler)

            node_cost_cols = [col for col in tasks.columns if col.startswith('Cost_')]

            node_features = torch.cat((node_features, torch.from_numpy(tasks[node_cost_cols].to_numpy())), dim=0)
            node_labels = torch.cat((node_labels, torch.from_numpy(tasks['task_label'].to_numpy())), dim=0)
                        
            edges_src = torch.cat((edges_src, torch.from_numpy(edges['src'].to_numpy() + n_tasks)), dim=0)
            edges_dst = torch.cat((edges_dst, torch.from_numpy(edges['dst'].to_numpy() + n_tasks)), dim=0)

            n_tasks += len(tasks)
            self.num_classes = network.number_of_nodes() # should be the same for all networks

            edge_cost_cols = [col for col in edges.columns if col.startswith('Cost_')]
            edge_features = torch.cat((edge_features, torch.from_numpy(edges[edge_cost_cols].to_numpy())), dim=0)

        for workflow, network in itertools.product(self.train_workflows, self.train_networks):
            process_pair(workflow, network)
            train_mask = torch.cat((train_mask, torch.tensor([True]*workflow.number_of_nodes())), dim=0)
            val_mask = torch.cat((val_mask, torch.tensor([False]*workflow.number_of_nodes())), dim=0)
            test_mask = torch.cat((test_mask, torch.tensor([False]*workflow.number_of_nodes())), dim=0)

        for workflow, network in itertools.product(self.val_workflows, self.val_networks):
            process_pair(workflow, network)
            train_mask = torch.cat((train_mask, torch.tensor([False]*workflow.number_of_nodes())), dim=0)
            val_mask = torch.cat((val_mask, torch.tensor([True]*workflow.number_of_nodes())), dim=0)
            test_mask = torch.cat((test_mask, torch.tensor([False]*workflow.number_of_nodes())), dim=0)
        
        for workflow, network in itertools.product(self.test_workflows, self.test_networks):
            process_pair(workflow, network)
            train_mask = torch.cat((train_mask, torch.tensor([False]*workflow.number_of_nodes())), dim=0)
            val_mask = torch.cat((val_mask, torch.tensor([False]*workflow.number_of_nodes())), dim=0)
            test_mask = torch.cat((test_mask, torch.tensor([True]*workflow.number_of_nodes())), dim=0)

        # normalize features
        max_feature = max(node_features.max(), edge_features.max())
        min_feature = min(node_features.min(), edge_features.min())
        node_features = (node_features - min_feature) / (max_feature - min_feature)
        edge_features = (edge_features - min_feature) / (max_feature - min_feature)

        self.graph = dgl.graph((edges_src, edges_dst), num_nodes=n_tasks)
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
