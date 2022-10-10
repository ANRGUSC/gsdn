from typing import List

import networkx as nx
import numpy as np

def gen_workflows(n_workflows: int, n_copies: int) -> List[nx.DiGraph]:
    workflows = []
    for i in range(n_workflows):
        workflow = nx.DiGraph()
        task_id_offset = 0
        for j in range(n_copies):
            name_dict = {
                task_name: i+task_id_offset for i, task_name in enumerate([
                    "Source","Copy","Tiler","Detect1",
                    "Detect2","Detect3","Feature merger",
                    "Graph Spiltter","Classify1","Classify2",
                    "Reco. Merge","Display"
                ])
            }
            task_id_offset += len(name_dict)
            dic_task_graph = dict()
            dic_task_graph[name_dict["Source"]] = [name_dict["Copy"]]
            dic_task_graph[name_dict["Copy"]] = [ name_dict["Tiler"],name_dict["Feature merger"],name_dict["Display"]]
            dic_task_graph[name_dict["Tiler"]] = [ name_dict["Detect1"],name_dict["Detect2"],name_dict["Detect3"] ]
            dic_task_graph[name_dict["Detect1"]] = [name_dict["Feature merger"]]
            dic_task_graph[name_dict["Detect2"]] = [name_dict["Feature merger"]]
            dic_task_graph[name_dict["Detect3"]] = [name_dict["Feature merger"]]
            dic_task_graph[name_dict["Feature merger"]] = [name_dict["Graph Spiltter"]]
            dic_task_graph[name_dict["Graph Spiltter"]] = [ name_dict["Classify1"],name_dict["Classify2"],name_dict["Reco. Merge"] ]
            dic_task_graph[name_dict["Classify1"]] = [name_dict["Reco. Merge"]]
            dic_task_graph[name_dict["Classify2"]] = [name_dict["Reco. Merge"]]
            dic_task_graph[name_dict["Reco. Merge"]] =[name_dict["Display"]]
            for task_id in name_dict.values():
                cost = max(max(0, np.random.normal(1, scale=1/2)), 2)
                workflow.add_node(task_id, cost=cost)
            for task_id in dic_task_graph:
                for child in dic_task_graph[task_id]:
                    data = 0 #1/2# max(max(0, np.random.normal(1, scale=1/2)), 2)
                    workflow.add_edge(task_id, child, data=data)
        workflows.append(workflow)
    return workflows
