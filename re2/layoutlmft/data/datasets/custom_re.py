import json
import os
import pickle
import pdb

import datasets
from datasets import DatasetDict, Dataset

from layoutlmft.models.layoutxlm.tokenization_layoutxlm import LayoutXLMTokenizer

from layoutlmft.data.utils import load_image, normalize_bbox
from layoutlmft.data.datasets.regions import generate_regions

logger = datasets.logging.get_logger(__name__)

 
dataset = DatasetDict()

def create_DatasetDict(folder_path, diverseForm = True):
    if(diverseForm):
        funsd_cache = "./layoutlmft/data/datasets/__pycache__/custom.pickle"
        if os.path.isfile(funsd_cache):
            with open(funsd_cache, "rb") as f:
                return pickle.load(f)
        train_path = os.path.join(folder_path, 'custom_train')
        test_path = os.path.join(folder_path, 'custom_test')
        dataset['validation'] = generate_examples(test_path)
        dataset['train'] = generate_examples(train_path)
        with open(funsd_cache, "wb") as f:
            pickle.dump(dataset, f)
    else:
        generate_regions(folder_path)
        dataset['test'] = generate_examples(folder_path)
    return dataset

def merge_partial_boxes(entities, index, item):
    min_x = pow(2,32)
    max_x = -1*pow(2,32)
    min_y = pow(2,32)
    max_y = -1*pow(2,32)
    for partial in range(entities['start'][index], entities['end'][index]):
        min_x = min(item['bbox'][partial][0], min_x)
        min_y = min(item['bbox'][partial][1], min_y)
        max_x = max(item['bbox'][partial][2], max_x)
        max_y = max(item['bbox'][partial][3], max_y)
    reg = [min_x,min_y,max_x,max_y]
    return reg

def IoU(box1, box2):

    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2

    if (x1 < x3 and x2 < x3) or (x3 < x1 and x4 < x1):
        return 0
    
    if (y1 < y3 and y2 < y3) or (y3 < y1 and y4 < y1):
        return 0
    
    if (y2 < y3) or (y1 > y4):
        return 0
    
    if (x2 < x3) or (x1 > x4):
        return 0
    
    if (x1 > x3 and x2 < x4) and (y1 > y3 and y2 < y4):
        return 0.5
    
    if (x1 < x3 and x2 > x4) and (y1 < y3 and y2 > y4):
        return 0.5

    
    x_inter1 = max(x1, x3)
    x_inter2 = min(x2, x4)
    y_inter1 = max(y1, y3)
    y_inter2 = min(y2, y4)

    width_inter = abs(x_inter1 - x_inter2)
    height_inter = abs(y_inter1 - y_inter2)
    area_inter = width_inter*height_inter

    

    area_1 = (x2-x1)*(y2-y1)
    area_2 = (x4-x3)*(y4-y3)

    area_union = area_1 + area_2 - area_inter

    iou = area_inter/area_union


    return iou

def generate_examples(filepath, diverseForm = True):
    logger.info("⏳ Generating examples from = %s", filepath)

    tokenizer = LayoutXLMTokenizer.from_pretrained("microsoft/layoutxlm-base")
    
    dataset = {'id':[],'input_ids':[],'bbox':[],'labels':[],'entities':[],'relations':[],'image':[],'region_ids':[]}

    ann_dir = os.path.join(filepath, "annotations")
    img_dir = os.path.join(filepath, "images")

    label2id = {'b-other':0,'i-other':0,'b-QUESTION':1,'b-ANSWER':2,'b-HEADER':3,'i-QUESTION':4,'i-ANSWER':5,'i-HEADER':6}
    
    for guid, file in enumerate(sorted(os.listdir(ann_dir))):
        word_list = []
        label_list = []
        bbox_list = []
        temp_id_list = []

        relation_list = []
        entities_dict = {'start':[],'end':[],'label':[], 'link_id':[], 'region_id':[], 'bbox':[]}
        relations_dict = {'head':[], 'tail':[], 'start_index':[],'end_index':[]}

        if(diverseForm):
            with open('xfun_custom.json') as json_file:
                        regions = json.load(json_file)
        else:
            with open('../Data/inference/regions.json') as json_file:
                regions = json.load(json_file)

        file_path = os.path.join(ann_dir, file)
        with open(file_path, "r", encoding="utf8") as f:
            data = json.load(f)
        image_path = os.path.join(img_dir, file)
        image_path = image_path.replace("json", "png")
        image, size = load_image(image_path)
        for item in data["form"]:
            #identify label of set of words
            words, label = item["words"], item["label"]
            words = [w for w in words if w["text"].strip() != ""]
            if len(words) == 0:
                continue
            #id used to keep track of links (pre tokenized, refers to entity number)
            id = item['id']

            # create list of relations for a doc
            if len(item['linking'])!=0:
                for i in item['linking']:
                    if i[2] == "QUESTION-ANSWER":
                        relation_list.append(i)

            if label == "other":
                for w in words:
                    word_list.append(w["text"])
                    label_list.append(label2id[f'i-{label}'])
                    temp_id_list.append(id)
                    bbox_list.append(normalize_bbox(w["box"], size))
            else:
                word_list.append(words[0]["text"])
                label_list.append(label2id[f'b-{label}'])
                temp_id_list.append(id)
                bbox_list.append(normalize_bbox(words[0]["box"], size))
                for w in words[1:]:
                    word_list.append(w["text"])
                    label_list.append(label2id[f'i-{label}'])
                    temp_id_list.append(id)
                    bbox_list.append(normalize_bbox(w["box"], size))
        
        embeddings = tokenizer(text=word_list, boxes=bbox_list, word_labels=label_list, truncation= False)
        temp_embed_for_id = tokenizer(text=word_list, boxes=bbox_list, word_labels=temp_id_list)

        ip_list = []
        bb_list = []
        lab_list = []
        
        is_first_entity = True
        entity_start = True
        for index in range(len(embeddings['input_ids'])):
            if(index>511):
                break
            ip_list.append(embeddings['input_ids'][index])
            bb_list.append(embeddings['bbox'][index])
            lab_list.append(embeddings['labels'][index])
            if(embeddings['labels'][index]!=-100 and is_first_entity and embeddings['labels'][index] not in {0,3,6}):
                entities_dict['start'].append(index)
                entities_dict['label'].append(embeddings['labels'][index])
                entities_dict['link_id'].append(temp_embed_for_id['labels'][index])
                is_first_entity = False
            elif(embeddings['labels'][index]!=-100 and is_first_entity!=True):
                if(embeddings['labels'][index]-3!=entities_dict['label'][-1]):
                    if entity_start:
                        entities_dict['end'].append(index)
                        entity_start = False
                    if (embeddings['labels'][index] not in {0,3,6}):
                        entities_dict['start'].append(index)
                        entity_start = True
                        entities_dict['label'].append(embeddings['labels'][index])
                        entities_dict['link_id'].append(temp_embed_for_id['labels'][index])
                    
            if(index==len(embeddings['input_ids'])-1):
                entities_dict['end'].append(index)
                entity_start = False
            index += 1

        if(len(entities_dict['start'])!=len(entities_dict['end'])):
            for i in range(len(embeddings['input_ids'])-1, -1, -1):
                if embeddings['input_ids'][i] != -100:
                    entities_dict['end'].append(i)
                    break

        final_rel_list = []
        [final_rel_list.append(x) for x in relation_list if x[0] not in final_rel_list]
        for rel in final_rel_list:
            try:
                head_id = entities_dict['link_id'].index(rel[0])
                tail_id = entities_dict['link_id'].index(rel[1])
                start_index = entities_dict['start'][entities_dict['link_id'].index(rel[0])]
                end_index = entities_dict['start'][entities_dict['link_id'].index(rel[1])]
                relations_dict['head'].append(head_id)
                relations_dict['tail'].append(tail_id)
                relations_dict['start_index'].append(start_index)
                relations_dict['end_index'].append(end_index)
            except:
                continue

        regions_doc = regions[file[:-4]+"png"]
        region_id = {}
        id_no = 0
        for en in range(len(entities_dict['start'])):

            entities_dict["bbox"].append(merge_partial_boxes(entities_dict, en , embeddings))

            bbox1 = embeddings['bbox'][entities_dict['start'][en]]
            flag = False
            temp_id = None
            if entities_dict['start'][en]+1 <entities_dict['end'][en]:
                bbox2 = embeddings['bbox'][entities_dict['start'][en]+1]
                flag = True
            bbox3 = embeddings['bbox'][entities_dict['end'][en]-1]

            for reg in regions_doc:
                ###completely inside
                reg = normalize_bbox(reg, size)
                if (bbox1[0] > reg[0] and bbox1[2] < reg[2]) and (bbox1[1] > reg[1] and bbox1[3] < reg[3]):
                    if tuple(reg) in region_id:
                        temp_id = region_id[tuple(reg)]
                    else:
                        region_id[tuple(reg)] = id_no
                        temp_id = id_no
                        id_no += 1
                    break
                elif flag and (bbox2[0] > reg[0] and bbox2[2] < reg[2]) and (bbox2[1] > reg[1] and bbox2[3] < reg[3]):
                    if tuple(reg) in region_id:
                        temp_id = region_id[tuple(reg)]
                    else:
                        region_id[tuple(reg)] = id_no
                        temp_id = id_no
                        id_no += 1
                    break
                elif (bbox3[0] > reg[0] and bbox3[2] < reg[2]) and (bbox3[1] > reg[1] and bbox3[3] < reg[3]):
                    if tuple(reg) in region_id:
                        temp_id = region_id[tuple(reg)]
                    else:
                        region_id[tuple(reg)] = id_no
                        temp_id = id_no
                        id_no += 1
                    break

                ###soft match###
                elif flag and (IoU(bbox1,reg) > 0 or IoU(bbox2,reg) > 0 or IoU(bbox3,reg) > 0):
                    if tuple(reg) in region_id:
                        temp_id = region_id[tuple(reg)]
                    else:
                        region_id[tuple(reg)] = id_no
                        temp_id = id_no
                        id_no += 1
                    break

            if(temp_id == None):
                reg = merge_partial_boxes(entities_dict, en, embeddings)

                if tuple(reg) in region_id:
                        temp_id = region_id[tuple(reg)]
                else:
                    region_id[tuple(reg)] = id_no
                    temp_id = id_no
                    id_no += 1
                
                entities_dict['region_id'].append(temp_id)
            else:
                entities_dict['region_id'].append(temp_id)

        regions_in_this_span = []
        for tup in region_id.keys():
            regions_in_this_span.append([region_id[tup],tup[0],tup[1],tup[2],tup[3]])

        sorted(regions_in_this_span, key=lambda x: x[0])
        
        dataset["id"].append(file[:-4]+"png")
        dataset["input_ids"].append(ip_list)
        dataset["bbox"].append(bb_list)
        dataset["labels"].append(lab_list)
        dataset["image"].append(image)
        del entities_dict['link_id']
        dataset["entities"].append(entities_dict)
        dataset["relations"].append(relations_dict)
        dataset['region_ids'].append(regions_in_this_span)

    dataset = Dataset.from_dict(dataset)
    return dataset


