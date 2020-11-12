import torch
import torch.nn as nn
import torch.nn.functional as F
from sinkhorn_knopp import sinkhorn_knopp as skp


class SinkhornKnopp(nn.Module):

    def forward(self, P: torch.Tensor, num_iter: int = 5):
        N = P.shape[0]
        for i in range(num_iter):
            P = P / P.sum(1).view((N, 1))
            P = P / P.sum(0).view((1, N))
        return P


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.graph_norm = SinkhornKnopp()

    def forward(self, input, adj):

        # Eq. 8
        h = torch.mm(input, self.W)  # matrix multiplication of the matrices

        # Eq. 11
        N = h.size()[0]
        e = torch.cat([h.repeat(1, N).view(N * N, -1),
                             h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        e = torch.matmul(e, self.a).squeeze(2)
        e = self.leakyrelu(e)
        e = torch.exp(e)

        # Eq. 10
        alfa = self.graph_norm(e * adj)

        # Dropout
        # alfa = nn.Dropout(p=0.2)(alfa)

        # Eq. 7
        h_prime = torch.matmul(alfa, h)

        return h_prime, alfa

