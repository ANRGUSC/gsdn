#!/usr/bin/env python3
"""
Run model script.
"""
import pathlib

import dill as pickle

from app import App
from dataset import WorkflowsDataset

thisdir = pathlib.Path(__file__).parent.resolve()

# Change These Parameters
MODEL = {
    "layer_type": "edGNNLayer",
    "layer_params": {
        "n_units": 64,
        "activation": "relu",
        "dropout": 0,
        "n_hidden_layers": 2
    }
}
HYPER_PARAMS = {
    'lr': 1e-3, 
    'n_epochs': 300, 
    'weight_decay': 5e-3, 
    'batch_size': 128
}

def main():
    data_path = thisdir / 'data'
    save_path = data_path.joinpath('model.pt')

    print('\n*** Set default saving/loading path to:', save_path)

    data: WorkflowsDataset = pickle.loads(data_path.joinpath('data.pkl').read_bytes())
    config_params = {
        **MODEL,
        "edge_dim": data.num_classes**2,
        "node_dim": data.num_classes
    }

    # 1. Training
    app = App(early_stopping=True)
    learning_config = {
        **HYPER_PARAMS,
        "cuda": False
    }
    print('\n*** Start training ***\n')
    app.train(
        data, config_params, learning_config, 
        save_path=save_path
    )

    # 2. Testing
    print('\n*** Start testing ***\n')
    app.test(data, save_path)


if __name__ == '__main__':
    main()
