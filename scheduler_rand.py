import random
from typing import Dict, Hashable
from matplotlib.style import available

import networkx as nx


def schedule(workflow: nx.DiGraph, network: nx.Graph) -> Dict[Hashable, Hashable]:
    """Schedule workflow on network using random algorithm
    
    Args:
        workflow (nx.DiGraph): Workflow graph
        network (nx.Graph): Network graph

    Returns:
        Dict[Hashable, Hashable]: Mapping from task to node
    """
    available_nodes = [
        node for node in network.nodes
        if network.nodes[node]['cpu'] > 1e-3
    ]
    return {task: random.choice(available_nodes) for task in workflow.nodes} 
