from typing import Callable, Dict, FrozenSet, Hashable, Iterable, List, Tuple, Union
import networkx as nx
import numpy as np 
from dataclasses import dataclass
import pathlib 
import matplotlib.pyplot as plt 
from copy import deepcopy 

thisdir = pathlib.Path(__file__).resolve().parent

@dataclass
class Task:
    locale: str
    name: str
    start: float
    end: float 
            
class Schedule:
    def __init__(self) -> None:
        pass

def schedule_task(schedule: List[Task], 
                  locale: str,
                  name: str, 
                  min_start_time: float, 
                  exec_time: float) -> Task:
    if not schedule or min_start_time + exec_time < schedule[0].start:
        new_task = Task(locale, name, min_start_time, min_start_time + exec_time)
        schedule.insert(0, new_task)
        return new_task
    
    for i, (left, right) in enumerate(zip(schedule, schedule[1:]), start=1):
        if min_start_time >= left.end and min_start_time + exec_time <= right.start:
            new_task = Task(locale, name, min_start_time, min_start_time + exec_time)
            schedule.insert(i, new_task)
            return new_task

    # No insert possible, append to end 
    start_time = max(min_start_time, schedule[-1].end)
    new_task = Task(locale, name, start_time, start_time + exec_time)
    schedule.append(new_task)
    return new_task

def heft(network: nx.Graph, 
         task_graph: nx.DiGraph,
         task_cost: Callable[[Hashable, Hashable], float]) -> Tuple[Dict[str, Task], Dict[str, List[Task]], Dict[str, List[Task]]]:
    paths: Dict[str, Dict[str, List[str]]] = nx.shortest_paths.shortest_path(network)
    comp_schedule: Dict[str, List[Task]] = {node: [] for node in network.nodes}
    comm_schedule: Dict[FrozenSet[str], List[Task]] = {frozenset(edge): [] for edge in network.edges}
    task_schedule: Dict[str, Task] = {}

    task_name: str
    for task_name in nx.topological_sort(task_graph):
        task_size = task_graph.nodes[task_name]["cost"] 
        parents = {parent for parent, _ in task_graph.in_edges(task_name)}

        min_finish_time = np.inf 
        best_comm_schedule = None 
        best_comp_schedule = None 
        best_task_schedule = None 
        for node in network.nodes:
            _comm_schedule = deepcopy(comm_schedule)
            _comp_schedule = deepcopy(comp_schedule)
            _task_schedule = deepcopy(task_schedule)

            # executes_on = task_graph.nodes[task_name]["executes_on"]
            # if executes_on is not None and node not in executes_on:
            #     continue

            node_speed = network.nodes[node]["cpu"] 
            min_arrival_time: float = 0 if not parents else np.inf
            for parent in parents:
                task_xfer_size = task_graph.edges[(parent, task_name)]["data"]
                shortest_path = paths[task_schedule[parent].locale][node]
                arrival_time = task_schedule[parent].end
                for src, dst in zip(shortest_path, shortest_path[1:]):
                    bandwidth: float = network.edges[(src, dst)]["bandwidth"]
                    comm_task = schedule_task(
                        schedule=_comm_schedule[frozenset((src, dst))],
                        locale=(src, dst),
                        name=(src, parent, dst),
                        min_start_time=arrival_time,
                        exec_time=task_xfer_size/bandwidth
                    )
                    _task_schedule[comm_task.name] = comm_task
                    arrival_time = comm_task.end

                min_arrival_time = min(min_arrival_time, arrival_time)
                
            comp_task = schedule_task(
                schedule=_comp_schedule[node],
                locale=node,
                name=task_name,
                min_start_time=min_arrival_time,
                exec_time=task_cost(task_name, node)
            )
            _task_schedule[comp_task.name] = comp_task
            finish_time = comp_task.end

            if finish_time < min_finish_time:
                min_finish_time = finish_time
                best_comm_schedule = _comm_schedule
                best_comp_schedule = _comp_schedule
                best_task_schedule = _task_schedule
        
        comm_schedule = best_comm_schedule 
        comp_schedule = best_comp_schedule 
        task_schedule = best_task_schedule 


    return task_schedule, comm_schedule, comp_schedule