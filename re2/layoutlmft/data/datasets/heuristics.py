import json 
import torch
import pdb

def find_bbox(lst, start_element):
    for i in range(len(lst)):
        if lst[i][0] == start_element:
            return lst[i][1:]
    return [0]*4

def check_lr_tb(head_bbox, tail_bbox):
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

def create_edge_feature(node_i, node_j, region_map):
    #determining heuristics links
    feature_links = []
    if(node_i[1] == node_j[1]):
        link = 7*[0]
        link[0] = 1
        link[1], link[2] = check_lr_tb(node_i[0], node_j[0])
        feature_links.append(link)
    else:
        link = 7*[0]
        link[3], link[4] = check_lr_tb(node_i[0], node_j[0])
        region_head_bbox = find_bbox(region_map, node_i[1])
        region_tail_bbox = find_bbox(region_map, node_j[1])
        link[5], link[6] = check_lr_tb(region_head_bbox, region_tail_bbox)
        feature_links.append(link)

    return feature_links

def create_graph(id_full, relations_full, entities_full, region_maps):
    sum = 0
    sum_g = 0
    graph_viz = {}
    graph_viz['actual'] = {}
    graph_viz['graph'] = {}
    flag = True
    for ids, relations, entities, region_map in zip(id_full, relations_full, entities_full, region_maps):
        questions_index = []
        questions_metadata = []
        answers_index = []
        answers_metadata = []
        for i, entity_index in enumerate(entities['start']):
            if(entities['label'][i]==1):
                questions_index.append(i)
                questions_metadata.append((entities['bbox'][i], entities['region_id'][i]))
            elif(entities['label'][i]==2):
                answers_index.append(i)
                answers_metadata.append((entities['bbox'][i], entities['region_id'][i]))

        #########create heuristic based graph#############
        if ids[:-2] not in graph_viz['graph'].keys():
            graph_viz['graph'][ids[:-2]] = []
            graph_viz['actual'][ids[:-2]] = []

        graph_relations = []

        pls_work = {}

        for i, head in enumerate(questions_index):
            for j, tail in enumerate(answers_index):
                feature_links = create_edge_feature(questions_metadata[i],answers_metadata[j],region_map)
                rel = str(head)+','+str(tail)
                pls_work[rel] = feature_links
                if any(feature_links):
                    graph_relations.append((head,tail))
                    graph_viz['graph'][ids[:-2]].append((questions_metadata[i][0], answers_metadata[j][0]))
        ################################################
        if(flag):
            out_file = open("pls_work_1.json", "w")
            json.dump(pls_work, out_file, indent = 6)
            flag = False
        ############create bipartite graph################
        bipartite = []
        for i, head in enumerate(questions_index):
            for j, tail in enumerate(answers_index):
                bipartite.append((head,tail))
        ##################################################

        actual_relations = []
        for i, head in enumerate(relations['head']):
            actual_relations.append((head,relations['tail'][i]))
            graph_viz['actual'][ids[:-2]].append((entities['bbox'][head],entities['bbox'][relations['tail'][i]]))
        pdb.set_trace()
        count = len([elem for elem in actual_relations if elem not in graph_relations])
        count_bip = len([elem for elem in actual_relations if elem not in bipartite])

        sum = sum + count
        sum_g = sum_g + len(actual_relations)

        print(f"relations not in graph {count} actual relations {len(actual_relations)} graph relations {len(graph_relations)}")

    out_file = open("graph_viz.json", "w")
    json.dump(graph_viz, out_file, indent = 6)


    #print(f"total missing:{sum}, total actual:{sum_g}")
    return
        