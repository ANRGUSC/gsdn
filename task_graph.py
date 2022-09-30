from base64 import b64encode, b64decode
from functools import partial
from typing import Any, Callable, Dict, List, Optional
import pickle

import inspect
import time
from wfcommons.wfchef.recipes.cycles.recipe import CyclesRecipe
from wfcommons.wfchef.recipes.montage import MontageRecipe
from wfcommons.wfchef.recipes.seismology import SeismologyRecipe
from wfcommons.wfchef.recipes.blast import BlastRecipe
from wfcommons.wfchef.recipes.bwa import BwaRecipe
from wfcommons.wfchef.recipes.epigenomics import EpigenomicsRecipe
from wfcommons.wfchef.recipes.srasearch import SrasearchRecipe
from wfcommons.wfchef.recipes.genome import GenomeRecipe
from wfcommons.wfchef.recipes.soykb import SoykbRecipe
from wfcommons.wfchef.utils import draw
import networkx as nx
import pathlib
import json

RECIPES = {
    "montage": MontageRecipe,
    "cycles": CyclesRecipe,
    "seismology": SeismologyRecipe,
    "blast": BlastRecipe,
    "bwa": BwaRecipe,
    "epigenomics": EpigenomicsRecipe,
    "srasearch": SrasearchRecipe,
    "genome": GenomeRecipe,
    "soykb": SoykbRecipe
}

RECIPE = "montage"
NUM_TASKS = 43

def deserialize(text: str) -> Any:
    return pickle.loads(b64decode(text))

def serialize(obj: Any) -> str:
    return b64encode(pickle.dumps(obj)).decode("utf-8")

class Task:
    def __init__(self, name: str, call: Callable, cost: float = 0.0) -> None:
        self.name = name
        self.call = call
        self.cost = cost

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.call(*args, **kwds)

class TaskGraph:
    def __init__(self) -> None:
        self.task_deps: Dict[Task, List[Task]] = {}
        self.task_names: Dict[str, Task] = {}

    def execute(self, name: str, *args, **kwargs) -> Any:
        return self.task_names[name](*args, **kwargs)

    def dependencies(self, name: str) -> List[str]:
        return [
            task.name for task in
            self.task_deps[self.task_names[name]]
        ]

    def task(self, *deps: Task) -> Callable:
        def _task(fun: Callable) -> Any:
            return self.add_task(fun, *deps)
        return _task

    def add_task(self, fun: Callable, *deps: Task, name: Optional[str] = None, cost: float = 0.0) -> Task:
        def _fun(*args, **kwargs) -> Any:
            args = [deserialize(arg) for arg in args]
            kwargs = {
                key: deserialize(value) 
                for key, value in kwargs.items()
            }
            return serialize(fun(*args, **kwargs))
        task = Task(fun.__name__ if not name else name, _fun, cost=cost)
        self.task_deps[task] = deps
        self.task_names[task.name] = task
        return task

    def __str__(self) -> str:
        return "\n".join([
            f"{task.name} <- [{', '.join([dep.name for dep in deps])}]"
            for task, deps in self.task_deps.items()
        ])

    def start_tasks(self) -> List[str]:
        return [
            task_name for task_name, task in self.task_names.items()
            if not self.task_deps[task]
        ]

    def end_tasks(self) -> List[str]:
        source_tasks = {
            dep.name for _, task in self.task_names.items() 
            for dep in self.task_deps[task]
        }
        return list(set(self.task_names.keys()) - source_tasks)

    def summary(self) -> Dict[str, List[str]]:
        return {
            task.name: [dep.name for dep in deps]
            for task, deps in self.task_deps.items()
        }

thisdir = pathlib.Path(__file__).resolve().parent
task_stats_path = pathlib.Path(inspect.getfile(RECIPES[RECIPE])).parent.joinpath("task_type_stats.json")
task_stats = json.loads(pathlib.Path(task_stats_path).read_text())

def fake_execute(name: str, execution_time: float, *args, **kwargs) -> str:
    print(f"{name}: SLEEPING FOR {execution_time} SECONDS")
    time.sleep(execution_time)
    return "Nothing"

def get_graph() -> TaskGraph:
    microstructures_path = pathlib.Path(inspect.getfile(RECIPES[RECIPE])).parent.joinpath("microstructures")
    workflow = None
    for graph_pickle in microstructures_path.glob("*/base_graph.pickle"):
        graph: nx.DiGraph = pickle.loads(graph_pickle.read_bytes())
        if workflow is None or graph.order() <= workflow.order():
            workflow = graph
        
    task_graph = TaskGraph()

    visited = {}
    final_nodes = []
    task_functions = {}
    
    queue = [task_name for task_name in workflow.nodes if workflow.in_degree(task_name) == 0]
    while queue:
        node = queue.pop(0)
        if node in visited:
            continue
        queue.extend(list(workflow.successors(node)))

        task_type = workflow.nodes[node]["type"]
        stats = task_stats.get(task_type, {"runtime": {"min": 0, "max": 0}})
        runtime = (stats["runtime"]["max"] - stats["runtime"]["min"])/2

        if task_type not in task_functions:
            task_functions[task_type] = partial(fake_execute, task_type, runtime/1000)
        
        deps = [
            visited[dep_name] for dep_name, _ in workflow.in_edges(node)
        ]
        visited[node] = task_graph.add_task(task_functions[task_type], *deps, name=node, cost=runtime)

        if workflow.out_degree(node) == 0:
            final_nodes.append(visited[node])
    
    return task_graph
    
if __name__ == "__main__":
    print(get_graph())
