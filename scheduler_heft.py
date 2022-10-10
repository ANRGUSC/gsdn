import pathlib
from dataclasses import dataclass
from typing import Dict, Hashable, List, Optional, Tuple

import networkx as nx
import numpy as np

thisdir = pathlib.Path(__file__).resolve().parent

@dataclass
class Task:
    node: str
    name: str
    start: Optional[float]
    end: Optional[float] 

def get_insert_loc(schedule: List[Task], 
                   min_start_time: float, 
                   exec_time: float) -> Tuple[int, float]:
    if not schedule or min_start_time + exec_time < schedule[0].start:
        return 0, min_start_time
    
    for i, (left, right) in enumerate(zip(schedule, schedule[1:]), start=1):
        if min_start_time >= left.end and min_start_time + exec_time <= right.start:
            return i, min_start_time
        elif min_start_time < left.end and left.end + exec_time <= right.start:
            return i, left.end

    return len(schedule), max(min_start_time, schedule[-1].end)

def heft_rank_sort(network: nx.Graph, task_graph: nx.DiGraph) -> List[Hashable]:
    rank = {}
    for task_name in reversed(list(nx.topological_sort(task_graph))):
        avg_comp = np.mean([
            task_graph.nodes[task_name]['cost'] / 
            network.nodes[node]['cpu'] for node in network.nodes
        ])
        max_comm = 0 if task_graph.out_degree(task_name) <= 0 else max(
            ( 
                rank.get(succ, 0) + 
                np.mean([
                    task_graph.edges[task_name, succ]['data'] /
                    network.edges[src, dst]['bandwidth'] for src, dst in network.edges
                ])
            )
            for succ in task_graph.successors(task_name)
        )
        rank[task_name] = avg_comp + max_comm
    return sorted(list(rank.keys()), key=rank.get, reverse=True)

# @lru_cache(maxsize=None)
def heft(network: nx.Graph, task_graph: nx.DiGraph) -> Tuple[Dict[str, Task], Dict[str, List[Task]]]:
    comp_schedule: Dict[str, List[Task]] = {node: [] for node in network.nodes}
    task_schedule: Dict[str, Task] = {}

    task_name: str
    for task_name in heft_rank_sort(network, task_graph):
        task_size = task_graph.nodes[task_name]["cost"] 

        min_finish_time = np.inf 
        best_node = None 
        for node in network.nodes: # Find the best node to run the task
            node_speed = network.nodes[node]["cpu"] 
            max_arrival_time: float = max(
                [
                    0.0, *[
                        task_schedule[parent].end + (
                            (
                                task_graph.edges[(parent, task_name)]["data"] / 
                                network.edges[(task_schedule[parent].node, node)]["bandwidth"]
                            )  if node != task_schedule[parent].node else 0
                        )
                        for parent in task_graph.predecessors(task_name)
                    ]
                ]
            )
                
            idx, start_time = get_insert_loc(comp_schedule[node], max_arrival_time, task_size / node_speed)
            
            finish_time = start_time + task_size / node_speed
            if finish_time < min_finish_time:
                min_finish_time = finish_time
                best_node = node, idx 
        
        task = Task(best_node[0], task_name, min_finish_time - task_size / network.nodes[best_node[0]]["cpu"], min_finish_time)
        comp_schedule[best_node[0]].insert(best_node[1], task)
        task_schedule[task_name] = task

    return task_schedule, comp_schedule

def schedule(workflow: nx.DiGraph, network: nx.Graph) -> Dict[Hashable, Hashable]:
    """Schedule workflow on network using HEFT algorithm
    
    Args:
        workflow (nx.DiGraph): Workflow graph
        network (nx.Graph): Network graph

        Returns:
            Dict[Hashable, Hashable]: Mapping from task to node
    """
    task_schedule, comp_schedule = heft(network, workflow)
    return {task.name: task.node for task in task_schedule.values()}
