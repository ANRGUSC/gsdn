import random
from typing import Dict, Hashable

import networkx as nx


def schedule(workflow: nx.DiGraph, network: nx.Graph) -> Dict[Hashable, Hashable]:
    """Schedule workflow on network using random algorithm
    
    Args:
        workflow (nx.DiGraph): Workflow graph
        network (nx.Graph): Network graph

    Returns:
        Dict[Hashable, Hashable]: Mapping from task to node
    """
    return {task: random.choice(list(network.nodes)) for task in workflow.nodes} 
