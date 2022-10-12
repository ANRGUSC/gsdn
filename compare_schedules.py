import pathlib
from pprint import pprint
import random
from simulate import run_workflow, load_dataset_workflows
from data_epigenomics import gen_workflows as gen_workflows_epigenomics
from scheduler_heft import Task, heft
from scheduler_gcn import schedule as schedule_gcn
from preprocess import gen_networks
import networkx as nx
from typing import Dict, Hashable, Optional
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from matplotlib import cm

thisdir = pathlib.Path(__file__).parent.resolve()


def draw_workflow(workflow: nx.DiGraph, 
                  schedule: Dict[Hashable, Hashable],
                  n_colors: Optional[int], 
                  ax: plt.Axes) -> None:
    """Draw workflow DAG
    
    """
    if n_colors is not None:
        cmap = cm.get_cmap('rainbow', n_colors)
        node_color = [cmap(schedule[task]) for task in workflow.nodes]
    else:
        node_color = ['black' for task in workflow.nodes]
    pos = nx.nx_agraph.pygraphviz_layout(workflow, prog='dot')
    nx.draw(
        workflow, pos, ax=ax, 
        with_labels=False, 
        node_color=node_color
    )


def main():
    num_nodes = 10
    num_networks = 50
    
    # workflows, *_ = load_dataset_workflows()
    # workflow = workflows[0]
    workflow = gen_workflows_epigenomics(n_workflows=1, n_copies=1)[0]
    network = random.choice(list(gen_networks(num_networks, num_nodes*2, num_nodes)))

    task_schedule_heft, comp_schedule_heft = heft(network, workflow)
    makespan_heft = max(task.end for task in task_schedule_heft.values())

    sched_gcn = schedule_gcn(workflow, network)
    nx.set_node_attributes(workflow, {task_name: node for task_name, node in sched_gcn.items()}, 'node')
    top_sort = list(nx.topological_sort(workflow))
    for task_name in reversed(top_sort):
        network.nodes[sched_gcn[task_name]].setdefault('tasks', []).append(task_name)
    makespan_gcn = run_workflow(workflow, network)

    comp_schedule_gcn = {}
    for task in top_sort:
        comp_schedule_gcn.setdefault(sched_gcn[task], []).append(
            Task(sched_gcn[task], task, workflow.nodes[task]['start_time'], workflow.nodes[task]['end_time'])
        )
    comp_schedule_gcn = {node: sorted(tasks, key=lambda task: task.start) for node, tasks in comp_schedule_gcn.items()}

    # Compare Schedules
    pprint(comp_schedule_gcn)
    pprint(comp_schedule_heft)

    print('GCN Makespan:', makespan_gcn)
    print('HEFT Makespan:', makespan_heft)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    sched_heft = {task.name: task.node for task in task_schedule_heft.values()}
    draw_workflow(workflow, sched_heft, num_nodes, ax1)
    draw_workflow(workflow, sched_gcn, num_nodes, ax2)
    savedir = thisdir / 'data' / 'plots' / 'images'
    savedir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(savedir / 'compare_schedules.png', dpi=300)

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    draw_workflow(workflow, sched_gcn, None, ax)
    fig.tight_layout()
    fig.savefig(savedir / 'task_graph.png', dpi=300)


if __name__ == '__main__':
    main()