import random

import torch
from scheduler_gcn import schedule as schedule_gcn, format_data, preprocess, Model
from scheduler_heft import schedule as schedule_heft

from data_epigenomics import gen_workflows as gen_workflows_epigenomics
from preprocess import gen_networks
from train import MODEL
import time
import pandas as pd
from joblib import Parallel, delayed
import pathlib 

import plotly.express as px

thisdir = pathlib.Path(__file__).parent.resolve()

def fake_schedule_gcn(workflow, network):
    t = time.time()
    dummy_schedule = lambda *_: {task: 0 for task in workflow.nodes}
    tasks, edges = preprocess(workflow, network, schedule=dummy_schedule)
    print(f"preprocess: {time.time() - t}")
    t = time.time()
    graph = format_data(tasks, edges)
    print(f"format_data: {time.time() - t}")
    t = time.time()
    config_params = {
        **MODEL,
        "edge_dim": network.number_of_nodes()**2,
        "node_dim": network.number_of_nodes()
    }
    model = Model(
        g=graph,
        config_params=config_params,
        n_classes=network.number_of_nodes(),
        is_cuda=False
    )
    print(f"fake_schedule_gcn: {time.time() - t}")
    with torch.no_grad():
        logits = model(None)
        preds = torch.argmax(logits, dim=1)
        return {task: preds[task] for task in workflow.nodes}

def main():
    def fun(num_nodes: int):
        workflow = gen_workflows_epigenomics(n_workflows=1, n_copies=5)[0]
        network = next(gen_networks(1, 1, num_nodes))

        t = time.time()
        sched_gcn = fake_schedule_gcn(workflow, network)
        t_gcn = time.time() - t
        t = time.time()
        sched_heft = schedule_heft(workflow, network)
        t_heft = time.time() - t

        print(f"num_nodes: {num_nodes}, t_gcn: {t_gcn}, t_heft: {t_heft}")

        return {
            'num_nodes': num_nodes,
            'GCN': t_gcn,
            'HEFT': t_heft,
        }

    rows = Parallel(n_jobs=-2)(delayed(fun)(num_nodes) for num_nodes in range(10, 200, 10))
    df = pd.DataFrame(rows)
    
    savedir = thisdir.joinpath('data', 'results')
    savedir.mkdir(exist_ok=True)
    df.to_csv(savedir.joinpath('runtime_experiments.csv'), index=False)

    pngdir = thisdir.joinpath('data', 'plots', 'images')
    pngdir.mkdir(exist_ok=True, parents=True)
    htmldir = thisdir.joinpath('data', 'plots', 'html')
    htmldir.mkdir(exist_ok=True, parents=True)

    fig = px.line(
        df, 
        x='num_nodes', 
        y=['GCN', 'HEFT'],
        template='plotly_white',
        labels={
            'num_nodes': '# Nodes',
        },
    )
    fig.write_image(pngdir.joinpath('runtime_experiments.png'))
    fig.write_html(htmldir.joinpath('runtime_experiments.html'))

if __name__ == '__main__':
    main()