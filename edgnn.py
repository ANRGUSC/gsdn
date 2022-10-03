"""
edGNN layer (add link to the paper)
"""
import math

import torch
import torch.nn as nn
from torch.nn import Linear


def init_weights(m):
    if isinstance(m, Linear):
        stdv = 1. / math.sqrt(m.weight.size(1))
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.uniform_(-stdv, stdv)

def reset(nn):
    def _reset(item):
        if hasattr(item, 'reset_parameters'):
            item.reset_parameters()

    if nn is not None:
        if hasattr(nn, 'children') and len(list(nn.children())) > 0:
            for item in nn.children():
                _reset(item)
        else:
            _reset(nn)

def reset_graph_features(g):
    keys = ['h_in', 'h_msg', 'm', 'hn_out']
    for key in keys:
        if key in g.ndata:
            del g.ndata[key]
    if 'he' in g.edata:
        del g.edata['he'] 


class edGNNLayer(nn.Module):
    def __init__(self,
                 g,
                 node_dim,
                 edge_dim,
                 out_feats,
                 activation=None,
                 dropout=None,
                 bias=None,
                 use_bn=False,
                 device=None):
        """
        edGNN Layer constructor.

        Args:
            g (dgl.DGLGraph): instance of DGLGraph defining the topology for message passing
            node_dim (int): node dimension
            edge_dim (int): edge dimension (if 1-hot, edge_dim)
            out_feats (int): hidden dimension
            activation: pyTorch functional defining the nonlinearity to use
            dropout (float or None): dropout probability
            bias (bool): if True, a bias term will be added before applying the activation
        """
        super(edGNNLayer, self).__init__()

        # 1. set parameters
        self.g = g
        self.node_dim = node_dim
        self.out_feats = out_feats
        self.activation = activation
        self.dropout = dropout
        self.edge_dim = edge_dim
        self.bias = bias
        self.use_bn = use_bn
        self.device = device

        # 2. create variables
        self._build_parameters()

        # 3. initialize variables
        self.apply(init_weights)
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.linear)

    def _build_parameters(self):
        """
        Build parameters and store them in a dictionary.
        The keys are the same keys of the node features to which we are applying the parameters.
        """
        input_dim = 2 * self.node_dim
        if self.edge_dim is not None:
            input_dim = input_dim + self.edge_dim

        self.linear = nn.Linear(input_dim, self.out_feats, bias=self.bias, device=self.device)

        # Dropout module
        if self.dropout:
            self.dropout = nn.Dropout(p=self.dropout)

        # Batch norm module
        if self.use_bn:
            self.bn = nn.BatchNorm1d(self.out_feats)

    def gnn_msg(self, edges):
        """
            If edge features: for each edge u->v, return as msg: MLP(concat([h_u, h_uv]))
        """
        if self.g.edata is not None:
            msg = torch.cat([edges.src['hn_in'],
                             edges.data['he']],
                            dim=1)
            if self.dropout:
                msg = self.dropout(msg)
        else:
            msg = edges.src['hn_in']
            if self.dropout:
                msg = self.dropout(msg)
        return {'m': msg}

    def gnn_reduce(self, nodes):
        accum = torch.sum((nodes.mailbox['m']), 1)
        return {'h_msg': accum}

    def node_update(self, nodes):
        h = torch.cat([nodes.data['hn_in'],
                       nodes.data['h_msg']],
                      dim=1)
        h = self.linear(h)

        if self.activation:
            h = self.activation(h)

        if self.dropout:
            h = self.dropout(h)

        if self.use_bn:
            h = self.bn(h)

        return {'hn_out': h}

    def forward(self, node_features, edge_features, g):

        if g is not None:
            self.g = g

        # 1. clean graph features
        reset_graph_features(self.g)

        # 2. set current iteration features
        self.g.ndata['hn_in'] = node_features
        self.g.edata['he'] = edge_features

        # 3. aggregate messages
        self.g.update_all(self.gnn_msg,
                          self.gnn_reduce,
                          self.node_update)

        h = self.g.ndata.pop('hn_out')
        return h
