import pathlib
from pprint import pprint
from simulate import run_workflow, load_dataset_workflows
from scheduler_heft import Task, heft
from scheduler_gcn import schedule as schedule_gcn
from preprocess import gen_networks
import networkx as nx
from typing import Dict, Hashable
import matplotlib.pyplot as plt
from matplotlib import cm

thisdir = pathlib.Path(__file__).parent.resolve()


def draw_workflow(workflow: nx.DiGraph, 
                  schedule: Dict[Hashable, Hashable],
                  n_colors: int, 
                  ax: plt.Axes) -> None:
    """Draw workflow DAG
    
    """
    cmap = cm.get_cmap('rainbow', n_colors)
    node_color = [cmap(schedule[task]) for task in workflow.nodes]
    pos = nx.nx_agraph.pygraphviz_layout(workflow, prog='dot')
    nx.draw(
        workflow, pos, ax=ax, 
        with_labels=False, 
        node_color=node_color,
        cmap=cmap
    )


def main():
    num_nodes = 10
    num_networks = 50
    
    workflows, *_ = load_dataset_workflows()
    workflow = workflows[0]
    network = next(gen_networks(num_networks, 1, num_nodes))

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
    plt.show()


if __name__ == '__main__':
    main()