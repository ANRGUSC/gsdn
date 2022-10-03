import time

import dgl
import numpy as np
import torch

from dataset import WorkflowsDataset
from early_stopping import EarlyStopping
from model import Model


def collate(samples):
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(labels).cuda() if labels[0].is_cuda else torch.tensor(labels)

class App:
    def __init__(self, early_stopping=True):
        if early_stopping:
            self.early_stopping = EarlyStopping(patience=100, verbose=True)

    def train(self, data: WorkflowsDataset, model_config, learning_config, save_path=''):
        loss_fcn = torch.nn.CrossEntropyLoss()

        labels = data.graph.ndata['labels']
        
        train_mask = data.graph.ndata['train_mask']
        val_mask = data.graph.ndata['val_mask']
        dur = []

        # create GNN model
        self.model = Model(
            g=data.graph,
            config_params=model_config,
            n_classes=data.num_classes,
            is_cuda=learning_config['cuda']
        )

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_config['lr'],
            weight_decay=learning_config['weight_decay']
        )

        for epoch in range(learning_config['n_epochs']):
            self.model.train()
            if epoch >= 3:
                t0 = time.time()
            # forward
            logits = self.model(None)
            loss = loss_fcn(logits[train_mask], labels[train_mask])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch >= 3:
                dur.append(time.time() - t0)

            val_acc, val_loss = self.model.eval(labels, val_mask)
            print("Epoch {:05d} | Time(s) {:.4f} | Train loss {:.4f} | Val accuracy {:.4f} | "
                    "Val loss {:.4f}".format(epoch,
                                            np.mean(dur),
                                            loss.item(),
                                            val_acc,
                                            val_loss))

            self.early_stopping(val_loss, self.model, save_path)

            if self.early_stopping.early_stop:
                print("Early stopping")
                break

    def test(self, data: WorkflowsDataset, load_path=''):
        try:
            print('*** Load pre-trained model ***')
            self.model.load_state_dict(torch.load(load_path))
        except ValueError as e:
            print('Error while loading the model.', e)

        test_mask = data.graph.ndata['test_mask']
        labels = data.graph.ndata['labels']
        acc, _ = self.model.eval(labels, test_mask)

        print("\nTest Accuracy {:.4f}".format(acc))

        return acc
