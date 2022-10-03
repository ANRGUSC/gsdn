from typing import List

import networkx as nx
import numpy as np


def gen_workflows(n_workflows: int) -> List[nx.DiGraph]:
    name_dict = {"Source":0,"Copy":1,"Tiler":2,"Detect1":3,"Detect2":4,"Detect3":5,"Feature merger":6,"Graph Spiltter":7,"Classify1":8,"Classify2":9
    ,"Reco. Merge":10,"Display":11}
    dic_task_graph = dict()
    dic_task_graph[name_dict["Source"]]=[name_dict["Copy"]]
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

    workflows = []
    for i in range(n_workflows):
        workflow = nx.DiGraph()
        for task_id in name_dict.values():
            cost = max(max(0, np.random.normal(1, scale=1/2)), 2)
            workflow.add_node(task_id, cost=cost)
        for task_id in dic_task_graph:
            for child in dic_task_graph[task_id]:
                data = 1 # max(max(0, np.random.normal(1, scale=1/2)), 2)
                workflow.add_edge(task_id, child, data=data)
        workflows.append(workflow)
    return workflows
