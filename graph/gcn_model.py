import torch.nn as nn
import torch.nn.functional as F
import torch

from graph.layers_gcn import GraphConvolution
from graph.layers_gcn import SinkhornKnopp


class EGNNC(nn.Module):
    """Dense version of GAT."""

    def __init__(self, nin, nhid, nheads, dropout, layers=2, residual=0):
        super(EGNNC, self).__init__()

        self.Gconv1 = nn.ModuleList([GraphConvolution(nin, nhid) for _ in range(nheads)])
        self.Gconv2 = nn.ModuleList([GraphConvolution(nhid * nheads, nhid) for _ in range(nheads)])
        self.graph_norm = SinkhornKnopp()
        self.act = torch.nn.ELU()
        self.dropout = dropout
        self.layers = layers
        self.residual = residual

    def forward_to_graph_conv_layer(self, x:torch.Tensor, adj:torch.Tensor,
                                    m:torch.nn.Module):
        x_res = []
        for i, m in enumerate(m):
            x_i = m(x, adj[i])
            x_res.append(x_i)
        x_res = torch.cat(x_res, 1)
        return x_res

    def forward(self, x0, adj0):

        if len(adj0.size()) > 2:
            adj0 = torch.stack([self.graph_norm(G) for G in adj0])
        else:
            adj0 = torch.unsqueeze(self.graph_norm(adj0), dim=0).repeat(2, 1, 1)
        x1 = self.forward_to_graph_conv_layer(x0, adj0, self.Gconv1)
        if self.residual:
            x0 = x0.repeat(1, adj0.shape[0])
            x1 = x1 + x0
        x1 = self.act(x1)
        if self.layers > 1:
            x1 = nn.Dropout(p=0.5)(x1)
            x2 = self.forward_to_graph_conv_layer(x1, adj0, self.Gconv2)
            if self.residual:
                x2 = x1 + x2
            x2 = self.act(x2)
            return x2
        else:
            return x1

