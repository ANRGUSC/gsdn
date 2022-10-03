import random
from typing import Callable, Dict, Generator

import networkx as nx
from wfcommons.common.file import FileLink
from wfcommons.wfchef.recipes.blast import BlastRecipe
from wfcommons.wfchef.recipes.bwa import BwaRecipe
from wfcommons.wfchef.recipes.cycles.recipe import CyclesRecipe
from wfcommons.wfchef.recipes.epigenomics import EpigenomicsRecipe
from wfcommons.wfchef.recipes.genome import GenomeRecipe
from wfcommons.wfchef.recipes.montage import MontageRecipe
from wfcommons.wfchef.recipes.seismology import SeismologyRecipe
from wfcommons.wfchef.recipes.soykb import SoykbRecipe
from wfcommons.wfchef.recipes.srasearch import SrasearchRecipe
from wfcommons.wfgen.abstract_recipe import WorkflowRecipe
from wfcommons.wfgen.generator import WorkflowGenerator

RECIPES: Dict[str, WorkflowRecipe] = {
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

def triangle_distribution(max_scale: int, min: int) -> int:
    assert max_scale > 1
    return int(min * random.triangular(1, max_scale))

def uniform_distribution(max_scale: int, min: int) -> int:
    assert max_scale > 1
    return int(min * (random.random() * (max_scale - 1) + 1))

def gen_workflows(n_workflows: int, 
                  recipe_name: str, 
                  get_order: Callable[[int], int]) -> Generator[nx.DiGraph, None, None]:
    workflows = []
    if recipe_name not in RECIPES:
        raise ValueError(f"Unknown recipe: {recipe_name}")
    Recipe = RECIPES[recipe_name]
    for i in range(n_workflows):
        recipe = Recipe.from_num_tasks(get_order(Recipe.min_tasks))
        generator = WorkflowGenerator(recipe)
        workflow = generator.build_workflow()

        new_workflow = nx.DiGraph()
        task_ids = {}
        # maxruntime = max(workflow.nodes[node]['task'].runtime for node in workflow.nodes)
        for i, node in enumerate(workflow.nodes):
            new_workflow.add_node(i, cost=workflow.nodes[node]['task'].runtime)
            task_ids[workflow.nodes[node]['task'].name] = i

        # maxsize = max(file.size for node in workflow.nodes for file in workflow.nodes[node]['task'].files)
        for src, dst in workflow.edges:
            src_outputs = {file.name for file in workflow.nodes[src]['task'].files if file.link == FileLink.OUTPUT}
            data_size = sum(
                file.size for file in workflow.nodes[dst]['task'].files
                if file.link == FileLink.INPUT and file.name in src_outputs
            )
            new_workflow.add_edge(
                task_ids[workflow.nodes[src]['task'].name],
                task_ids[workflow.nodes[dst]['task'].name],
                data=data_size
            )

        max_data_size = max(new_workflow.edges[edge]['data'] for edge in new_workflow.edges)
        max_task_cost = max(new_workflow.nodes[node]['cost'] for node in new_workflow.nodes)

        for edge in new_workflow.edges:
            new_workflow.edges[edge]['data'] = new_workflow.edges[edge]['data'] / max_data_size * max_task_cost

        print(f'Generated workflow with {len(new_workflow.nodes)} tasks and {len(new_workflow.edges)} edges')
        # draw(new_workflow, save=thisdir.joinpath(recipe_name))
        yield new_workflow
