"""
Model Interface
"""
import copy

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F

from edgnn import edGNNLayer

ACTIVATIONS = {
    'relu': F.relu,
    'sigmoid': torch.sigmoid,
    'tanh': torch.tanh,
    'leaky_relu': F.leaky_relu,
    'elu': F.elu,
    'selu': F.selu,
    'gelu': F.gelu,
}

def layer_build_args(node_dim, edge_dim, n_classes, layer_params):
    """
    Generator of layer arguments
    Args:
        layer_params (dict): Refer to constructor
    """
    if isinstance(layer_params['n_units'], list):
        for v in layer_params.values():
            assert isinstance(v, list), "Expected list because n_units is specified as list!"
            assert len(v) == len(layer_params['n_units']), "Expected same number of elements in lists!"
        params = copy.deepcopy(layer_params)
        n_layers = len(layer_params['n_units'])
    else:
        params = dict()
        n_layers = layer_params['n_hidden_layers']
        for k, v in layer_params.items():
            if k != 'n_hidden_layers':
                params[k] = [layer_params[k]]*n_layers

    n_units = params.pop('n_units')
    activations = params.pop('activation')
    kwargs = [dict(zip(params, t)) for t in zip(*params.values())]
    if len(kwargs) == 0:
        kwargs = [{}]*n_layers

    if n_layers == 0:
        yield node_dim, edge_dim, n_classes, None, {k: None for k in params.keys()}
    else:
        # input layer
        yield node_dim, edge_dim, n_units[0], ACTIVATIONS[activations[0]], kwargs[0]

        # hidden layers
        for i in range(n_layers - 1):
            yield n_units[i], edge_dim, n_units[i+1], ACTIVATIONS[activations[i+1]], kwargs[i]

        yield n_units[-1], edge_dim, n_classes, None, kwargs[-1]


class Model(nn.Module):

    def __init__(self, g, config_params, n_classes=None, is_cuda=False):
        """
        Instantiate a graph neural network.

        Args:
            g (DGLGraph): a preprocessed DGLGraph
            config_json (str): path to a configuration JSON file. It must contain the following fields: 
                               "layer_type", and "layer_params". 
                               The "layer_params" should be a (nested) dictionary containing at least the fields 
                               "n_units" and "activation". "layer_params" should contain other fields that corresponds
                               to keyword arguments of the concrete layers (refer to the layers implementation).
                               The name of these additional fields should be the same as the keyword args names.
                               The parameters in "layer_params" should either be lists with the same number of elements,
                               or single values. If single values are specified, then a "n_hidden_layers" (integer) 
                               field is expected.
                               The fields "n_input" and "n_classes" are required if not specified 
        """
        super(Model, self).__init__()

        self.is_cuda = is_cuda
        self.config_params = config_params
        self.n_classes = n_classes
        self.g = g

        self.build_model()

    def build_model(self):
        # Build NN
        self.layers = nn.ModuleList()
        layer_params = self.config_params['layer_params']

        self.node_dim = self.config_params['node_dim']
        self.edge_dim = self.config_params['edge_dim']

        # basic tests
        assert (self.n_classes is not None)

        # build and append layers
        # print('\n*** Building model ***')
        for node_dim, edge_dim, n_out, act, kwargs in layer_build_args(self.node_dim, self.edge_dim, self.n_classes,
                                                                       layer_params):
            # print('* Building new layer with args:', node_dim, edge_dim, n_out, act, kwargs)
            self.layers.append(edGNNLayer(self.g, node_dim, edge_dim, n_out, act, device=None if not self.is_cuda else torch.cuda.current_device(), **kwargs))
        # print('*** Model successfully built ***\n')

    def forward(self, g):
        if g is not None:
            g.set_n_initializer(dgl.init.zero_initializer)
            g.set_e_initializer(dgl.init.zero_initializer)
            self.g = g

        # 1. Build node features
        node_features = self.g.ndata['node_features'].float()
        node_features = node_features.cuda() if self.is_cuda else node_features

        # 2. Build edge features
        edge_features = self.g.edata['edge_features'].float()
        edge_features = edge_features.cuda() if self.is_cuda else edge_features

        # 3. Iterate over each layer
        for layer_idx, layer in enumerate(self.layers):
            if layer_idx == 0:
                h = layer(node_features, edge_features, self.g)
                self.g.ndata['h_0'] = h
            else:
                h = layer(h, edge_features, self.g)
                key = 'h_' + str(layer_idx)
                self.g.ndata[key] = h

        return h

    def eval(self, labels, mask):
        super().eval()
        loss_fcn = torch.nn.CrossEntropyLoss()
        with torch.no_grad():
            logits = self(None)
            logits = logits[mask]
            labels = labels[mask]
            loss = loss_fcn(logits, labels)
            _, indices = torch.max(logits, dim=1)
            correct = torch.sum(indices == labels)
            return correct.item() * 1.0 / len(labels), loss

