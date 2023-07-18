import dgl
import torch as th
from egat import EGATConv
import pdb

num_nodes = 45 
num_node_feats, num_edge_feats = 20, 20
num_attn_heads = 1


contacts = th.rand((num_nodes, num_nodes)) #random distance matrix
adj = contacts > 0.5 #binarize to obtain adjecency matrix
u, v = th.nonzero(adj, as_tuple=True) #edges list
pdb.set_trace()
graph = dgl.graph((u,v))    

node_feats = th.rand((num_nodes, num_node_feats)) 
edge_feats = th.rand((graph.number_of_edges(), num_edge_feats))

#use as regular torch/dgl layer work similar as GATConv from dgl library
egat = EGATConv(in_node_feats=num_node_feats,
                in_edge_feats=num_edge_feats,
                out_node_feats=10,
                out_edge_feats=10,
                num_heads=3)

new_node_feats, new_edge_feats = egat(graph, node_feats, edge_feats)
#new_node_feats.shape = (*, num_heads, out_node_feats)
#new_eode_feats.shape = (*, num_heads, out_edge_feats)