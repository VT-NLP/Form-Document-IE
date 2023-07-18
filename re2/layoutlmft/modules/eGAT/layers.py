import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, edge_features, out_features, batch_size, dropout, alpha, concat=True, residual = True):
        #in_features = 768, out_features = 384, batch_size = 2 ...
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.dropout_ffn = 0.1
        self.in_features = in_features
        self.out_features = out_features
        self.batch_size = batch_size
        self.alpha = alpha
        self.concat = concat
        self.edge_features = edge_features
        self.residual = residual

        projection_feature = nn.Sequential(
            nn.Linear(edge_features, out_features),
            nn.ReLU(),
            #nn.Dropout(self.dropout_ffn),
            nn.Linear(out_features, out_features // 2),
            nn.ReLU(),
            #nn.Dropout(self.dropout_ffn),
        )

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))# [768,768]
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))# [1, 1536]
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.ffn = copy.deepcopy(projection_feature)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj, edge, x):
        Wh = torch.matmul(h, self.W) # h.shape: (2, 64, 768), Wh.shape: (2, 64, 768)
        e, edge_weights = self._prepare_attentional_mechanism_input(Wh, edge)
        e = e/(torch.abs(torch.max(e, dim=-1, keepdim=True)[0])+1e-6)
        if torch.isnan(e).any():
            print(":(")
        zero_vec = -1e6*torch.ones_like(e, dtype=torch.float32)
        #e = torch.tensor(e, dtype=torch.float32)
        e = e.to(torch.float32)
        attention = torch.where(adj > 0, e, zero_vec) #condition, if true, otherwise
        attention = F.softmax(attention, dim=2, dtype=torch.float32)
        #attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.bmm(attention, Wh) + x# [2,64,768]

        if self.concat:
            return F.elu(h_prime), edge_weights
        else:
            return h_prime, edge_weights

    def _prepare_attentional_mechanism_input(self, Wh, edge):
        # Wh1 = torch.bmm(Wh, self.a[:Wh.shape[0],:self.out_features, :]) #[2, 64, 1]
        # Wh2 = torch.bmm(Wh, self.a[:Wh.shape[0],self.out_features:, :]) #[2, 64, 1]
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # Wh1 = torch.matmul(Wh, self.a)
        # Wh2 = torch.transpose(torch.matmul(Wh, self.a),1,2)
        Wh2 = torch.transpose(Wh2,1,2)
        # broadcast add
        e = Wh1 + Wh2 #[2, 64, 64]

        e = self.leakyrelu(e)

        dim0 = edge.shape[0]
        dim1 = edge.shape[1]
        dim2 = edge.shape[2]
        # Reshape the edge matrix to (batch_size * 64, edge_features)
        edge = edge.view(dim0 * dim1 * dim2, self.edge_features)
        # Pass the edge matrix through the sequential network
        edge = self.ffn(edge)
        # Reshape the output to (batch_size, 64, in_features)
        edge = edge.view(dim0, dim1, dim2, -1)
        e = e.unsqueeze(dim=-1)

        att = e * (1+edge)
        att = torch.sum(att, dim = -1)
        return att, edge

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
