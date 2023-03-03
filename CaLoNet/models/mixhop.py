import argparse
import copy
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange

import dgl
import dgl.function as fn
#from dgl.data import CiteseerGraphDataset, CoraGraphDataset, PubmedGraphDataset

class MixHopConv(nn.Module):
    r"""
    Description
    -----------
    MixHop Graph Convolutional layer from paper `MixHop: Higher-Order Graph Convolutional Architecturesvia Sparsified Neighborhood Mixing
     <https://arxiv.org/pdf/1905.00067.pdf>`__.
    .. math::
        H^{(i+1)} =\underset{j \in P}{\Bigg\Vert} \sigma\left(\widehat{A}^j H^{(i)} W_j^{(i)}\right),
    where :math:`\widehat{A}` denotes the symmetrically normalized adjacencymatrix with self-connections,
    :math:`D_{ii} = \sum_{j=0} \widehat{A}_{ij}` its diagonal degree matrix,
    :math:`W_j^{(i)}` denotes the trainable weight matrix of different MixHop layers.
    Parameters
    ----------
    in_dim : int
        Input feature size. i.e, the number of dimensions of :math:`H^{(i)}`.
    out_dim : int
        Output feature size for each power.
    p: list
        List of powers of adjacency matrix. Defaults: ``[0, 1, 2]``.
    dropout: float, optional
        Dropout rate on node features. Defaults: ``0``.
    activation: callable activation function/layer or None, optional
        If not None, applies an activation function to the updated node features.
        Default: ``None``.
    batchnorm: bool, optional
        If True, use batch normalization. Defaults: ``False``.
    """

    def __init__(
        self,
        in_dim,
        out_dim,
        p=[0, 1, 2],
        dropout=0,
        activation=None,
        batchnorm=False,
    ):
        super(MixHopConv, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.p = p
        self.activation = activation
        self.batchnorm = batchnorm

        # define dropout layer
        self.dropout = nn.Dropout(dropout)

        # define batch norm layer
        if self.batchnorm:
            self.bn = nn.BatchNorm1d(out_dim * len(p))

        # define weight dict for each power j
        self.weights = nn.ModuleDict(
            {str(j): nn.Linear(in_dim, out_dim, bias=False) for j in p}
        )

    def forward(self, graph, feats):
        with graph.local_scope():
            # assume that the graphs are undirected and graph.in_degrees() is the same as graph.out_degrees()
            degs = graph.in_degrees().float().clamp(min=1)
            norm = torch.pow(degs, -0.5).to(feats.device).unsqueeze(1)
            max_j = max(self.p) + 1
            outputs = []
            for j in range(max_j):

                if j in self.p:
                    output = self.weights[str(j)](feats)
                    outputs.append(output)

                feats = feats * norm
                graph.ndata["h"] = feats
                graph.update_all(fn.copy_u("h", "m"), fn.sum("m", "h"))
                feats = graph.ndata.pop("h")
                feats = feats * norm

            final = torch.cat(outputs, dim=1)

            if self.batchnorm:
                final = self.bn(final)

            if self.activation is not None:
                final = self.activation(final)

            final = self.dropout(final)

            return final