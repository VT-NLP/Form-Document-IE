import copy
import pdb
import json
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import wandb

class ConstraintLoss(torch.nn.Module):
    def __init__(self):
        super(ConstraintLoss, self).__init__()
    
    def forward(self, logits, relations):
        answer_map = {}
        for i, answer in enumerate(relations['tail']):
            answer_map[i] = [j for j, x in enumerate(relations['tail']) if x == answer and i!=j]
        
        loss = 0
        for i, logit in enumerate(logits):
            if relations['label'][i] and len(logits[answer_map[i]])!=0:
                prob_score_tp = torch.softmax(logit, dim = -1) + 1e-5
                prob_score_fp = torch.softmax(logits[answer_map[i]], dim=-1)
                log_logit = torch.log(prob_score_tp[1])
                one_minus_logits = -1*torch.log(1-prob_score_fp[:,1] + 1e-5)
                mean = torch.mean(one_minus_logits)
                loss += torch.abs(log_logit + mean)
                # if torch.isnan(loss):
                #     pdb.set_trace()
        return loss
             

class BiaffineAttention(torch.nn.Module):
    """Implements a biaffine attention operator for binary relation classification.

    PyTorch implementation of the biaffine attention operator from "End-to-end neural relation
    extraction using deep biaffine attention" (https://arxiv.org/abs/1812.11275) which can be used
    as a classifier for binary relation classification.

    Args:
        in_features (int): The size of the feature dimension of the inputs.
        out_features (int): The size of the feature dimension of the output.

    Shape:
        - x_1: `(N, *, in_features)` where `N` is the batch dimension and `*` means any number of
          additional dimensisons.
        - x_2: `(N, *, in_features)`, where `N` is the batch dimension and `*` means any number of
          additional dimensions.
        - Output: `(N, *, out_features)`, where `N` is the batch dimension and `*` means any number
            of additional dimensions.

    Examples:
        >>> batch_size, in_features, out_features = 32, 100, 4
        >>> biaffine_attention = BiaffineAttention(in_features, out_features)
        >>> x_1 = torch.randn(batch_size, in_features)
        >>> x_2 = torch.randn(batch_size, in_features)
        >>> output = biaffine_attention(x_1, x_2)
        >>> print(output.size())
        torch.Size([32, 4])
    """

    def __init__(self, in_features, out_features):
        super(BiaffineAttention, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.bilinear = torch.nn.Bilinear(in_features, in_features, out_features, bias=False)
        self.linear = torch.nn.Linear(2 * in_features, out_features, bias=True)
        self.linear_feature = torch.nn.Linear(in_features, out_features, bias=False)

        self.reset_parameters()

    def forward(self, x_1, x_2):
        return self.bilinear(x_1, x_2) + self.linear(torch.cat((x_1, x_2), dim=-1)) #+ self.linear_feature(x_3)

    def reset_parameters(self):
        self.bilinear.reset_parameters()
        self.linear.reset_parameters()


class REDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.entity_emb = nn.Embedding(3, config.hidden_size // 2, scale_grad_by_freq=True)

        projection = nn.Sequential(
            #GAT (node)
            #nn.Linear(2*(config.hidden_size) + config.hidden_size // 2, config.hidden_size),
            #eGAT (edge+node)
            nn.Linear(3*(config.hidden_size), config.hidden_size),
            #eGAT (edge)
            #nn.Linear(2*(config.hidden_size), config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
        )

        projection_feature = nn.Sequential(
            nn.Linear(2 , config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
        )

        self.ffnn_head = copy.deepcopy(projection)
        self.ffnn_tail = copy.deepcopy(projection)
        self.ffn_feature = copy.deepcopy(projection_feature)
        self.rel_classifier = BiaffineAttention(config.hidden_size // 2, 2)
        self.loss_fct = CrossEntropyLoss()
        self.loss_constraint = ConstraintLoss()

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
    
    def find_bbox(self, lst, start_element):
        for i in range(len(lst)):
            if lst[i][0] == start_element:
                return lst[i][1:]
        return [0]*4 
    
    def heuristic_links_per_pair(self,entities, region_ids, i, j):
        entities_region_id = entities["region_id"]
        region_head = entities_region_id[i]
        region_tail = entities_region_id[j]

        link = 7*[0]

        if(region_head == region_tail):
            link[0] = 1
            link[1], link[2] = self.check_lr_tb(entities["bbox"][i],entities["bbox"][i])
        else:
            link = 7*[0]
            link[3], link[4] = self.check_lr_tb(entities["bbox"][i],entities["bbox"][i])
            region_head_bbox = self.find_bbox(region_ids, region_head)
            region_tail_bbox = self.find_bbox(region_ids, region_tail)
            link[5], link[6] = self.check_lr_tb(region_head_bbox, region_tail_bbox)        
        
        return link

    def heuristic_links_per_doc(self,entities, region_ids, head_entities, tail_entities, device):
        if("region_id" in entities):
            entities_region_ids = torch.tensor(entities["region_id"], device=device)
            region_head = entities_region_ids[head_entities]
            region_tail = entities_region_ids[tail_entities]

            feature_links = []

            for i in range(len(head_entities)):
                if(region_head[i] == region_tail[i]):
                    link = 7*[0]
                    link[0] = 1
                    link[1], link[2] = self.check_lr_tb(entities["bbox"][head_entities[i]],entities["bbox"][tail_entities[i]])
                    feature_links.append(link)
                else:
                    link = 7*[0]
                    link[3], link[4] = self.check_lr_tb(entities["bbox"][head_entities[i]],entities["bbox"][tail_entities[i]])
                    region_head_bbox = self.find_bbox(region_ids, region_head[i])
                    region_tail_bbox = self.find_bbox(region_ids, region_tail[i])
                    link[5], link[6] = self.check_lr_tb(region_head_bbox, region_tail_bbox)
                    feature_links.append(link)

            feature_links = torch.tensor(feature_links, device=device, dtype=torch.float16)
        else:
            feature_links = torch.tensor([7*[0]]*len(head_entities), device=device, dtype=torch.float16)
        
        return feature_links
    

    def build_relation(self, relations, entities, region_ids):
        batch_size = len(relations)
        new_relations = []
        for b in range(batch_size):
            if len(entities[b]["start"]) <= 2:
                entities[b] = {"end": [1, 1], "label": [0, 0], "start": [0, 0]}
            
            all_possible_relations = set(
                [
                    (i, j)
                    for i in range(len(entities[b]["label"]))
                    for j in range(len(entities[b]["label"]))
                    if entities[b]["label"][i] == 1 and entities[b]["label"][j] == 2
                ]
            )
            if len(all_possible_relations) == 0:
                all_possible_relations = set([(0, 1)])
            positive_relations = set(list(zip(relations[b]["head"], relations[b]["tail"])))
            negative_relations = all_possible_relations - positive_relations
            positive_relations = set([i for i in positive_relations if i in all_possible_relations])
            reordered_relations = list(positive_relations) + list(negative_relations)
            relation_per_doc = {"head": [], "tail": [], "label": []}
            relation_per_doc["head"] = [i[0] for i in reordered_relations]
            relation_per_doc["tail"] = [i[1] for i in reordered_relations]
            relation_per_doc["label"] = [1] * len(positive_relations) + [0] * (
                len(reordered_relations) - len(positive_relations)
            )
            assert len(relation_per_doc["head"]) != 0
            new_relations.append(relation_per_doc)
        return new_relations, entities

    def get_predicted_relations(self, logits, relations, entities,):
        #one answer has only one question constraint
        pred_relations = []
        a_q_pair = {}
        for i, pred_label in enumerate(logits.argmax(-1)):
            if pred_label != 1:
                continue
            rel = {}
            prob = logits[i][pred_label]
            rel["head_id"] = relations["head"][i]
            rel["head"] = (entities["start"][rel["head_id"]], entities["end"][rel["head_id"]])
            rel["head_type"] = entities["label"][rel["head_id"]]

            rel["tail_id"] = relations["tail"][i]
            #new q_a pair with already existing ans, check prob. If new prob is less, don't append rel
            # if(rel["tail_id"] in a_q_pair):
            #     if(a_q_pair[rel["tail_id"]][0] < prob):
            #         a_q_pair[rel["tail_id"]] = (prob, rel["head_id"])
            #         #find the prev pred rel where the tail_id is the same
            #         for i in range(len(pred_relations)):
            #             if(pred_relations[i]["tail_id"] == rel["tail_id"]):
            #                 pred_relations[i]["head_id"] = rel["head_id"]
            #     else:
            #         continue
            # else:
            #     a_q_pair[rel["tail_id"]] = (prob,rel["head_id"])

            rel["tail"] = (entities["start"][rel["tail_id"]], entities["end"][rel["tail_id"]])
            rel["tail_type"] = entities["label"][rel["tail_id"]]
            rel["type"] = 1
            pred_relations.append(rel)
        return pred_relations

    def forward(self, hidden_states, node_states, edge_states,entities, relations, region_ids):
        batch_size, max_n_words, context_dim = hidden_states.size()
        device = hidden_states.device
        #get all possible relations 
        relations, entities = self.build_relation(relations, entities, region_ids)
        loss = 0
        all_pred_relations = []
        for b in range(batch_size):
            head_entities = torch.tensor(relations[b]["head"], device=device)
            tail_entities = torch.tensor(relations[b]["tail"], device=device)
            head_index = torch.tensor(relations[b]["head"], device=device)#get indices of all entity present in entities dict
            tail_index = torch.tensor(relations[b]["tail"], device=device)
            relation_labels = torch.tensor(relations[b]["label"], device=device)
            entities_start_index = torch.tensor(entities[b]["start"], device=device)
            entities_labels = torch.tensor(entities[b]["label"], device=device)
            head_label = entities_labels[head_index]
            head_label_repr = self.entity_emb(head_label)
            tail_label = entities_labels[tail_index]
            tail_label_repr = self.entity_emb(tail_label)

            head_index_residual = entities_start_index[head_index]
            tail_index_residual = entities_start_index[tail_index]

            #determining heuristics links
            #feature_links = self.heuristic_links_per_doc(entities[b],region_ids[b],head_entities, tail_entities, device)
            
            ########without region#############
            # for i in range(len(head_entities)):
            #     link = 2*[0]
            #     link[0], link[1] = self.check_lr_tb(entities[b]["bbox"][head_entities[i]],entities[b]["bbox"][tail_entities[i]])
            #     feature_links.append(link)
            
            # feature_links = torch.tensor(feature_links, device=device, dtype=torch.float16)

            #feature = self.ffn_feature(feature_links)

            #concat with eGAT states (edge+node)
            head_repr = torch.cat(
                (hidden_states[b][head_index_residual],node_states[b][head_index], edge_states[b][head_index, tail_index, :], head_label_repr),
                dim=-1,
            )
            tail_repr = torch.cat(
                (hidden_states[b][tail_index_residual],node_states[b][tail_index], edge_states[b][head_index, tail_index, :], tail_label_repr),
                dim=-1,
            )

            #concat with GAT (node)
            # head_repr = torch.cat(
            #     (hidden_states[b][head_index_residual],node_states[b][head_index], head_label_repr),
            #     dim=-1,
            # )
            # tail_repr = torch.cat(
            #     (hidden_states[b][tail_index_residual],node_states[b][tail_index], tail_label_repr),
            #     dim=-1,
            # )

            #concat with eGAT (edge)
            # head_repr = torch.cat(
            #     (hidden_states[b][head_index_residual],edge_states[b][head_index, tail_index, :], head_label_repr),
            #     dim=-1,
            # )
            # tail_repr = torch.cat(
            #     (hidden_states[b][tail_index_residual],edge_states[b][head_index, tail_index, :], tail_label_repr),
            #     dim=-1,
            # )

            #Baseline
            # head_repr = torch.cat(
            #     (hidden_states[b][head_index_residual], head_label_repr),
            #     dim=-1,
            # )
            # tail_repr = torch.cat(
            #     (hidden_states[b][tail_index_residual], tail_label_repr),
            #     dim=-1,
            # )
            heads = self.ffnn_head(head_repr)
            tails = self.ffnn_tail(tail_repr)

            logits = self.rel_classifier(heads, tails)
            loss1 = self.loss_fct(logits, relation_labels)
            loss += loss1
            loss2 = self.loss_constraint(logits, relations[b])
            loss += 0.02*loss2
            pred_relations = self.get_predicted_relations(logits, relations[b], entities[b])
            all_pred_relations.append(pred_relations)
        return loss, all_pred_relations
