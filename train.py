#!/usr/bin/env python3
"""
Run model script.
"""
import pathlib
import pprint

import torch
GPU = 0
torch.cuda.set_device(torch.device('cuda:{}'.format(GPU)))

from core.app import App
from core.data.constants import (GRAPH, LABELS, N_CLASSES,
                                 TEST_MASK, TRAIN_MASK, VAL_MASK)
from core.data.utils import complete_path, load_pickle
from core.models.constants import NODE_CLASSIFICATION
from utils.inits import to_cuda
from utils.io import print_graph_stats

thisdir = pathlib.Path(__file__).parent.resolve()

# Change These Parameters
MODEL = {
    "layer_type": "edGNNLayer",
    "layer_params": {
        "n_units": 64,
        "activation": "relu",
        "dropout": 0,
        "n_hidden_layers": 15
    }
}
HYPER_PARAMS = {
    'lr': 1e-3, 
    'n_epochs': 500, 
    'weight_decay': 5e-4, 
    'batch_size': 256
}

def load_data():
    folder = thisdir.joinpath('data')
    data = {
        GRAPH: load_pickle(complete_path(folder, GRAPH)),
        N_CLASSES: load_pickle(complete_path(folder, N_CLASSES))
    }

    for k in [LABELS, TRAIN_MASK, TEST_MASK, VAL_MASK]:
        data[k] = torch.load(complete_path(folder, k))

    return data

def main():
    if GPU < 0:
        cuda = False
    else:
        cuda = True
        print('GPU: {}'.format(GPU))
        torch.cuda.set_device(GPU)

    save_path = thisdir.joinpath('model.pt')
    print('\n*** Set default saving/loading path to:', save_path)

    data = load_data()
    if cuda:
        data[GRAPH] = data[GRAPH].to(torch.device('cuda:{}'.format(GPU)))

    if cuda:
        data[TRAIN_MASK] = data[TRAIN_MASK].cuda()
        data[VAL_MASK] = data[VAL_MASK].cuda()
        data[TEST_MASK] = data[TEST_MASK].cuda()
        data[LABELS] = data[LABELS].cuda()

    # data = to_cuda(data) if cuda else data
    mode = NODE_CLASSIFICATION

    print_graph_stats(data[GRAPH])

    config_params = {
        **MODEL,
        "edge_dim": data[N_CLASSES]**2,
        "node_dim": data[N_CLASSES]
    }

    pprint.pprint({
        **config_params,
        **dict(
            g=data[GRAPH],
            config_params=config_params,
            n_classes=data[N_CLASSES],
            is_cuda=cuda,
            mode=NODE_CLASSIFICATION
        )
    })

    # # 1. Training
    app = App(early_stopping=True)
    learning_config = {
        **HYPER_PARAMS,
        "cuda": cuda
    }
    print('\n*** Start training ***\n')
    app.train(
        data, config_params, learning_config, 
        save_path=save_path, 
        mode=mode
    )

    # 2. Testing
    print('\n*** Start testing ***\n')
    app.test(data, save_path, mode=mode)

    # # 3. Delete model
    # remove_model(save_path)

    # # Save model
    # torch.save(app.model, thisdir.joinpath('model'))


if __name__ == '__main__':
    main()
