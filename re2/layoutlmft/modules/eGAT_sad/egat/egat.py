"""Torch modules for graph attention networks with fully valuable edges (EGAT)."""
# pylint: disable= no-member, arguments-differ, invalid-name
import torch
from torch import nn
from torch.nn import init
import dgl
import pdb

# pylint: enable=W0235
class EGATConv(nn.Module):
    r"""
    
    Description
    -----------
    Apply Graph Attention Layer over input graph. EGAT is an extension
    of regular `Graph Attention Network <https://arxiv.org/pdf/1710.10903.pdf>`__ 
    handling edge features, detailed description is available in
    `Rossmann-Toolbox <https://pubmed.ncbi.nlm.nih.gov/34571541/>`__ (see supplementary data).
     The difference appears in the method how unnormalized attention scores :math:`e_{ij}`
     are obtain:
        
    .. math::
        e_{ij} &= \vec{F} (f_{ij}^{\prime})
        f_{ij}^{\prim} &= \mathrm{LeakyReLU}\left(A [ h_{i} \| f_{ij} \| h_{j}]\right)
    where :math:`f_{ij}^{\prim}` are edge features, :math:`\mathrm{A}` is weight matrix and 
    :math: `\vec{F}` is weight vector. After that resulting node features 
    :math:`h_{i}^{\prim}` are updated in the same way as in regular GAT. 
   
    Parameters
    ----------
    in_node_feats : int
        Input node feature size :math:`h_{i}`.
    in_edge_feats : int
        Input edge feature size :math:`f_{ij}`.
    out_node_feats : int
        Output nodes feature size.
    out_edge_feats : int
        Output edge feature size.
    num_heads : int
        Number of attention heads.
        
    Examples
    ----------
    >>> import dgl
    >>> import torch as th
    >>> from dgl.nn import EGATConv
    >>> 
    >>> num_nodes, num_edges = 8, 30
    >>>#define connections
    >>> u, v = th.randint(num_nodes, num_edges), th.randint(num_nodes, num_edges) 
    >>> graph = dgl.graph((u,v))    
    >>> node_feats = th.rand((num_nodes, 20)) 
    >>> edge_feats = th.rand((num_edges, 12))
    >>> egat = EGATConv(in_node_feats=20,
                          in_edge_feats=12,
                          out_node_feats=15,
                          out_edge_feats=10,
                          num_heads=3)
    >>> #forward pass                    
    >>> new_node_feats, new_edge_feats = egat(graph, node_feats, edge_feats)
    >>> new_node_feats.shape, new_edge_feats.shape
    ((8, 3, 12), (30, 3, 10))
    """
    def __init__(self,
                 in_node_feats,
                 in_edge_feats,
                 out_node_feats,
                 out_edge_feats,
                 num_heads,
                 **kw_args):
        
        super().__init__()
        self._num_heads = num_heads
        self._out_node_feats = out_node_feats
        self._out_edge_feats = out_edge_feats
        self.fc_nodes = nn.Linear(in_node_feats, out_node_feats*num_heads, bias=True)
        self.fc_edges = nn.Linear(in_edge_feats + 2*in_node_feats, out_edge_feats*num_heads, bias=False)
        self.fc_attn = nn.Linear(out_edge_feats, num_heads, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        """
        Reinitialize learnable parameters.
        """
        gain = init.calculate_gain('relu')
        init.xavier_normal_(self.fc_nodes.weight, gain=gain)
        init.xavier_normal_(self.fc_edges.weight, gain=gain)
        init.xavier_normal_(self.fc_attn.weight, gain=gain)

    def edge_attention(self, edges):
        #extract features
        h_src = edges.src['h']
        h_dst = edges.dst['h']
        f = edges.data['f']
        #stack h_i | f_ij | h_j
        stack = torch.cat([h_src, f, h_dst], dim=-1)
        # apply FC and activation
        f_out = self.fc_edges(stack)
        f_out = nn.functional.leaky_relu(f_out)
        f_out = f_out.view(-1, self._num_heads, self._out_edge_feats)
        # apply FC to reduce edge_feats to scalar
        a = self.fc_attn(f_out).sum(-1).unsqueeze(-1)

        return {'a': a, 'f' : f_out}

    def message_func(self, edges):
        return {'h': edges.src['h'], 'a': edges.data['a']}

    def reduce_func(self, nodes):
        alpha = nn.functional.softmax(nodes.mailbox['a'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['h'], dim=1)
        return {'h': h}
    
    def create_edge_feature(self, node_i, node_j, region_ids):
        #determining heuristics links
        feature_links = []
        if(node_i[3] == node_j[3]):
            link = 7*[0]
            link[0] = 1
            link[1], link[2] = self.check_lr_tb(node_i[2], node_j[2])
            feature_links.append(link)
        else:
            link = 7*[0]
            link[3], link[4] = self.check_lr_tb(node_i[2], node_j[2])
            region_head_bbox = self.find_bbox(region_ids, node_i[3])
            region_tail_bbox = self.find_bbox(region_ids, node_j[3])
            link[5], link[6] = self.check_lr_tb(region_head_bbox, region_tail_bbox)
            feature_links.append(link)

        return feature_links

    def create_graph(self, batch_sequence_output, region_ids, batch_entities):
        batch_graph = []
        for entities, embeddings in zip(batch_entities, batch_sequence_output):
            if 'region_id' not in entities:
                return batch_graph 
            entity_spans = list(zip(entities['start'], entities['end'], entities['bbox'], entities['region_id'], entities['label']))
            entity_embeddings = []
            for start, end, bbox, id, label in entity_spans:
                average = sum(embeddings[start:end])/len(embeddings[start:end])
                entity_embeddings.append((average, bbox, id, label))
            num_nodes = len(entity_embeddings)

            matrix = torch.rand((num_nodes, num_nodes))
            adj = matrix >= 0
            u, v = torch.nonzero(adj, as_tuple=True)
            graph = dgl.graph((u,v))    
            graph.remove_self_loop()
            graph.to(batch_sequence_output[0][0].device)
            node_features = []
            for i, embed in enumerate(entity_embeddings):
                node_features.append(embed[0])
            node_features = torch.stack(node_features)
            graph.ndata['h'] = node_features
            for i, node_i in enumerate(entity_embeddings):
                for j in range(i+1, num_nodes):
                    node_j = entity_embeddings[j]
                    if node_i['bbox']!=node_j['bbox']:
                        feature_links = self.create_edge_feature(node_i,node_j,region_ids)
                        graph.edges[i,i+1].data['f'] = torch.tensor(feature_links, device=batch_sequence_output.device, dtype=torch.float16)
            batch_graph.append(graph)
        return batch_graph
    
    def forward(self, sequence_output, region_ids, entities):
        r"""
        Compute new node and edge features.
        Parameters
        ----------
        graph : DGLGraph
            The graph.
        nfeats : torch.Tensor
            The input node feature of shape :math:`(*, D_{in})`
            where:
                :math:`D_{in}` is size of input node feature,
                :math:`*` is the number of nodes.
        efeats: torch.Tensor
             The input edge feature of shape :math:`(*, F_{in})`
             where:
                 :math:`F_{in}` is size of input node feauture,
                 :math:`*` is the number of edges.
       
            
        Returns
        -------
        pair of torch.Tensor
            node output features followed by edge output features
            The node output feature of shape :math:`(*, H, D_{out})` 
            The edge output feature of shape :math:`(*, H, F_{out})`
            where:
                :math:`H` is the number of heads,
                :math:`D_{out}` is size of output node feature,
                :math:`F_{out}` is size of output edge feature.            
        """

        batch_graph = self.create_graph(sequence_output, region_ids, entities)
        node_feature_dim = sequence_output[0][0].shape[0]
        edge_feature_dim  = 7

        output = []

        for graph in batch_graph:
            with graph.local_scope():
            ##TODO allow node src and dst feats
                graph.apply_edges(self.edge_attention)

                nfeats_ = self.fc_nodes(node_feature_dim)
                nfeats_ = nfeats_.view(-1, self._num_heads, self._out_node_feats)

                graph.ndata['h'] = nfeats_
                graph.update_all(message_func = self.message_func,
                            reduce_func = self.reduce_func)

                output.append(graph.ndata.pop('h'), graph.edata.pop('f'))
        return output