from typing import List

import networkx as nx
import numpy as np

task_graph = {
    'task1': ['task2a', 'task2b', 'task2c', 'task2d'],
    'task2a': ['task3a'], 'task2b': ['task3b'], 'task2c': ['task3c'], 'task2d': ['task3d'],
    'task3a': ['task4a'], 'task3b': ['task4b'], 'task3c': ['task4c'], 'task3d': ['task4d'],
    'task4a': ['task5a'], 'task4b': ['task5b'], 'task4c': ['task5c'], 'task4d': ['task5d'],
    'task5a': ['task7'], 'task5b': ['task7'], 'task5c': ['task7'], 'task5d': ['task7'],
    'task7': ['task8'],
    'task8': [],
}

def gen_workflows(n_workflows: int, n_copies: int) -> List[nx.DiGraph]:
    workflows = []
    for i in range(n_workflows):
        workflow = nx.DiGraph()
        task_id_offset = 0
        for j in range(n_copies):
            name_dict = {
                task_name: i+task_id_offset 
                for i, task_name in enumerate(sorted(task_graph.keys()))
            }
            dic_task_graph = {
                name_dict[task_name]: [name_dict[child] for child in task_graph[task_name]]
                for task_name in task_graph
            }
            for task_id in name_dict.values():
                cost = 2*min(max(0, np.random.normal(1, scale=1/2)), 2)
                workflow.add_node(task_id, cost=cost)
            for task_id in dic_task_graph:
                for child in dic_task_graph[task_id]:
                    data = 1 # max(max(0, np.random.normal(1, scale=1/2)), 2)
                    workflow.add_edge(task_id, child, data=data)
        workflows.append(workflow)
    return workflows
