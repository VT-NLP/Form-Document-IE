import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import GraphAttentionLayer
import pdb

class GAT(nn.Module):
    #nfeat=768, efeat=7, batch_size= 2, graph_size=64, nhid=768, dropout=0.6, alpha=0.2, nheads=8
    def __init__(self, nfeat, efeat, batch_size, graph_size, nhid, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout
        self.graph_size = graph_size
        self.nfeat = nfeat
        self.batch_size = batch_size

        self.attentions1 = [GraphAttentionLayer(nfeat, efeat, nhid, batch_size, dropout=dropout, alpha=alpha, concat=True, residual=True) for _ in range(nheads)]
        self.attentions2 = [GraphAttentionLayer(nhid, nhid // 2, nhid, batch_size, dropout=dropout, alpha=alpha, concat=True, residual=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions1):
            self.add_module('attention1_{}'.format(i), attention)
        for i, attention in enumerate(self.attentions2):
            self.add_module('attention2_{}'.format(i), attention)
    
    def find_bbox(self, lst, start_element):
        for i in range(len(lst)):
            try:
                if lst[i][0] == start_element:
                    return lst[i][1:]
            except:
                print("AAAAAA")
                return [0]*4 
        return [0]*4 

    def check_lr_tb(self, head_bbox, tail_bbox):
        lr = 0
        tb = 0
        if((tail_bbox[1]>=head_bbox[1] and tail_bbox[1]<=head_bbox[3]) or (tail_bbox[3]>=head_bbox[1] and tail_bbox[3]<=head_bbox[3])):
            lr = 1 #lr
        elif((head_bbox[1]>=tail_bbox[1] and head_bbox[1]<=tail_bbox[3]) or (head_bbox[3]>=tail_bbox[1] and head_bbox[3]<=tail_bbox[3])):
            lr = 1 #lr
        if((tail_bbox[0]>=head_bbox[0] and tail_bbox[0]<=head_bbox[2]) or (tail_bbox[2]>=head_bbox[0] and tail_bbox[2]<=head_bbox[2])):
            tb = 1 #tb
        elif((head_bbox[0]>=tail_bbox[0] and head_bbox[0]<=tail_bbox[2]) or (head_bbox[2]>=tail_bbox[0] and head_bbox[2]<=tail_bbox[2])):
            tb = 1 #tb
        return lr, tb

    def create_edge_feature(self, node_i, node_j, region_ids, regions=True, entities=True):
        #determining heuristics links
        if(regions and entities):
            feature_links = []
            if(node_i[1] == node_j[1]):
                link = 7*[0]
                link[0] = 1
                link[1], link[2] = self.check_lr_tb(node_i[0], node_j[0])
                feature_links.append(link)
            else:
                link = 7*[0]
                link[3], link[4] = self.check_lr_tb(node_i[0], node_j[0])
                region_head_bbox = self.find_bbox(region_ids, node_i[1])
                region_tail_bbox = self.find_bbox(region_ids, node_j[1])
                link[5], link[6] = self.check_lr_tb(region_head_bbox, region_tail_bbox)
                feature_links.append(link)

        elif(entities):
            feature_links = []
            link = 2*[0]
            link[0], link[1] = self.check_lr_tb(node_i[0], node_j[0])
            feature_links.append(link)

        else:
            feature_links = []
            link = 3*[0]
            if(node_i[1] == node_j[1]):
                link[0] = 1
            region_head_bbox = self.find_bbox(region_ids, node_i[1])
            region_tail_bbox = self.find_bbox(region_ids, node_j[1])
            link[1], link[2] = self.check_lr_tb(region_head_bbox, region_tail_bbox)
            feature_links.append(link)

        return feature_links

    def create_graph(self, batch_sequence_output, batch_region_ids, batch_entities):
        graph_size = self.graph_size
        device = batch_sequence_output.device
        batch_node = []
        batch_edge = []
        batch_adj = []
        for entities, embeddings, region_ids in zip(batch_entities, batch_sequence_output, batch_region_ids):
            matrix = torch.zeros(graph_size, graph_size, device=device)
            #######################################################################
            edge_features_matrix = torch.zeros(graph_size,graph_size,7, device=device)
            #######################################################################
            if 'region_id' not in entities:
                batch_edge.append([self.nfeat*[0]])
                batch_adj.append(matrix)
                batch_node.append(edge_features_matrix)
                print("WEEEWOOOWEEEWOOOO PROBLEM ZONE")
            else: 
                entity_spans = list(zip(entities['start'], entities['end'], entities['bbox'], entities['region_id'], entities['label']))
                node_features = []
                node_metadata = []
                for start, end, bbox, id, label in entity_spans:
                    # average = sum(embeddings[start:end])/len(embeddings[start:end])
                    # node_features.append(average)
                    node_features.append(embeddings[start])
                    node_metadata.append((bbox, id, label))
                num_nodes = len(node_features)
                for i in range(num_nodes, graph_size):
                    node_features.append(torch.zeros(768, device=device))

                #matrix[:num_nodes,:num_nodes] = torch.ones(num_nodes,num_nodes, device=device)
                
                ### creating bipartite graph ########################
                ###GAT1###
                questions_index = []
                answers_index = []
                for index, ent in enumerate(node_metadata):
                    if(ent[2] == 1):
                        questions_index.append(index)
                    else:
                        answers_index.append(index)
                for i in questions_index:
                    for j in answers_index:
                        matrix[i,j] = torch.tensor(1, device=device, dtype=torch.float32)
                        matrix[j,i] = torch.tensor(1, device=device, dtype=torch.float32)
                # ################################################

                #########create heuristic based graph#############
                ###GAT2###
                # questions_index = []
                # answers_index = []
                # for index, ent in enumerate(node_metadata):
                #     if(ent[2] == 1):
                #         questions_index.append((index,ent))
                #     else:
                #         answers_index.append((index,ent))
                # for i in questions_index:
                #     for j in answers_index:
                #         feature_links = self.create_edge_feature(i[1],j[1],region_ids)
                #         if any(feature_links):
                #             matrix[i[0],j[0]] = torch.tensor(1, device=device, dtype=torch.float32)
                #             matrix[j[0],i[0]] = torch.tensor(1, device=device, dtype=torch.float32)
                ################################################
                
                #########create heuristic based graph with A-A link#############
                ###GAT3###
                # questions_index = []
                # answers_index = []
                # for index, ent in enumerate(node_metadata):
                #     if(ent[2] == 1):
                #         questions_index.append((index,ent))
                #     else:
                #         answers_index.append((index,ent))
                # for i in questions_index:
                #     for j in answers_index:
                #         matrix[i[0],j[0]] = torch.tensor(1, device=device, dtype=torch.float32)
                #         matrix[j[0],i[0]] = torch.tensor(1, device=device, dtype=torch.float32)
                # for i in answers_index:
                #     for j in answers_index:
                #         if(i!=j):
                #             feature_links = self.create_edge_feature(i[1], j[1], region_ids)
                #             if any(feature_links):
                #                 matrix[i[0],j[0]] = torch.tensor(1, device=device, dtype=torch.float32)
                #                 matrix[j[0],i[0]] = torch.tensor(1, device=device, dtype=torch.float32)
                ################################################

                for i, node_i in enumerate(node_metadata):
                    for j in range(i+1, num_nodes):
                        node_j = node_metadata[j]
                        ############################################################################33
                        feature_links = self.create_edge_feature(node_i,node_j,region_ids, regions=True, entities=True)
                        ##################################################################################
                        edge_features_matrix[i][j] = torch.tensor(feature_links, device=device, dtype=torch.float16)
                        edge_features_matrix[j][i] = torch.tensor(feature_links, device=device, dtype=torch.float16)
                node_features = torch.stack(node_features)
                batch_node.append(node_features)
                batch_adj.append(matrix)
                batch_edge.append(edge_features_matrix)
        
        batch_node = torch.stack(batch_node)
        batch_adj = torch.stack(batch_adj)
        batch_edge = torch.stack(batch_edge)
        return batch_node, batch_adj, batch_edge


    def forward(self, sequence_output, region_ids, entities):            
        x, adj, e = self.create_graph(sequence_output, region_ids, entities)
        #x -> no_nodes * features [2,64,768]
        #adj -> no_nodes*no_nodes [2,64, 64]
        #e -> link vector [2,64,64,7]
        x = F.dropout(x, 0.1, training=self.training)

        outputs = [att(x, adj, e, x) for att in self.attentions1]

        x_prime = torch.stack([output[0] for output in outputs], dim=1)  # shape: (batch_size, num_heads, output_size)
        e_prime = torch.stack([output[1] for output in outputs], dim=1)  # shape: (batch_size, num_heads, output_size)

        x_prime = torch.mean(x_prime, dim=1)
        e_prime = torch.mean(e_prime, dim=1)

        outputs = [att(x_prime, adj, e_prime, x) for att in self.attentions2]

        x_prime = torch.stack([output[0] for output in outputs], dim=1)  # shape: (batch_size, num_heads, output_size)
        e_prime = torch.stack([output[1] for output in outputs], dim=1)  # shape: (batch_size, num_heads, output_size)

        x_prime = torch.mean(x_prime, dim=1)
        e_prime = torch.mean(e_prime, dim=1)

        return x_prime, e_prime
