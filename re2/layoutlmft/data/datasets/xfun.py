# Lint as: python3
import json
import logging
import os

import datasets
import pdb
import json

from layoutlmft.data.utils import load_image, merge_bbox, normalize_bbox, simplify_bbox
from transformers import AutoTokenizer


_URL = "https://github.com/doc-analysis/XFUN/releases/download/v1.0/"

_LANG = ["zh", "de", "es", "fr", "en", "it", "ja", "pt"]
logger = logging.getLogger(__name__)


class XFUNConfig(datasets.BuilderConfig):
    """BuilderConfig for XFUN."""

    def __init__(self, lang, additional_langs=None, **kwargs):
        """
        Args:
            lang: string, language for the input text
            **kwargs: keyword arguments forwarded to super.
        """
        super(XFUNConfig, self).__init__(**kwargs)
        self.lang = lang
        self.additional_langs = additional_langs


class XFUN(datasets.GeneratorBasedBuilder):
    """XFUN dataset."""

    BUILDER_CONFIGS = [XFUNConfig(name=f"xfun.{lang}", lang=lang) for lang in _LANG]

    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "input_ids": datasets.Sequence(datasets.Value("int64")),
                    "bbox": datasets.Sequence(datasets.Sequence(datasets.Value("int64"))),
                    "labels": datasets.Sequence(datasets.Value("int64")),
                    "image": datasets.Array3D(shape=(3, 224, 224), dtype="uint8"),
                    "entities": datasets.Sequence(
                        {
                            "start": datasets.Value("int64"),
                            "end": datasets.Value("int64"),
                            "label": datasets.ClassLabel(names=["HEADER", "QUESTION", "ANSWER"]),
                            "region_id": datasets.Value("int64"),
                            "bbox": datasets.Sequence(datasets.Value("int64"))
                        }
                    ),
                    "relations": datasets.Sequence(
                        {
                            "head": datasets.Value("int64"),
                            "tail": datasets.Value("int64"),
                            "start_index": datasets.Value("int64"),
                            "end_index": datasets.Value("int64"),
                        }
                    ),
                    "region_ids": datasets.Sequence(
                            datasets.Sequence(datasets.Value("int64"))
                    ),
                    "input_words": datasets.Sequence(
                            datasets.Value("string")
                    )
                }
            ),
            supervised_keys=None,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        urls_to_download_target = {
            "train": [f"{_URL}{self.config.lang}.train.json", f"{_URL}{self.config.lang}.train.zip"],
            "val": [f"{_URL}{self.config.lang}.val.json", f"{_URL}{self.config.lang}.val.zip"],
            # "test": [f"{_URL}{self.config.lang}.test.json", f"{_URL}{self.config.lang}.test.zip"],
        }
        #downloaded_files = dl_manager.download_and_extract(urls_to_download)
        train_files_for_many_langs = []
        val_files_for_many_langs = []
        # test_files_for_many_langs = [downloaded_files["test"]]
        if self.config.additional_langs:
            additional_langs = self.config.additional_langs.split("+")
            if "all" in additional_langs:
                additional_langs = [lang for lang in _LANG if lang != self.config.lang]
            for lang in additional_langs:
                if lang != "en":
                    urls_to_download = {"train": [f"{_URL}{lang}.train.json", f"{_URL}{lang}.train.zip"]}
                    additional_downloaded_files = dl_manager.download_and_extract(urls_to_download)
                    train_files_for_many_langs.append(additional_downloaded_files["train"])
        
        downloaded_files = dl_manager.download_and_extract(urls_to_download_target)
        train_files_for_many_langs.append(downloaded_files["train"])
        val_files_for_many_langs.append(downloaded_files["val"])
        
        logger.info(f"Training on {self.config.lang} with additional langs({self.config.additional_langs})")
        logger.info(f"Evaluating on {self.config.lang}")
        logger.info(f"Testing on {self.config.lang}")
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepaths": train_files_for_many_langs}),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION, gen_kwargs={"filepaths": val_files_for_many_langs}
            ),
            # datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepaths": test_files_for_many_langs}),
        ]
    
    def _generate_examples(self, filepaths):

        def merge_partial_boxes(entities, item):
            min_x = pow(2,32)
            max_x = -1*pow(2,32)
            min_y = pow(2,32)
            max_y = -1*pow(2,32)
            for partial in range(entities['start'], entities['end']):
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

        final_dict = {}

        for filepath in filepaths:
            logger.info("Generating examples from = %s", filepath)
            with open(filepath[0], "r", encoding="utf-8") as f:
                data = json.load(f)

            with open(f'xfun_{self.config.lang}.json') as json_file:
                    regions = json.load(json_file)

            if self.config.additional_langs:
                additional_langs = self.config.additional_langs.split("+")
                for lang in additional_langs:
                    with open(f'xfun_{lang}.json') as json_file:
                        regions.update(json.load(json_file))

            for doc in data["documents"]:
                doc["img"]["fpath"] = os.path.join(filepath[1], doc["img"]["fname"])
                image, size = load_image(doc["img"]["fpath"])
                document = doc["document"]
                tokenized_doc = {"input_ids": [], "bbox": [], "labels": []}
                entities = []
                relations = []
                id2label = {}
                entity_id_to_index_map = {}
                empty_entity = set()
                for line in document:
                    if len(line["text"]) == 0:
                        empty_entity.add(line["id"])
                        continue
                    id2label[line["id"]] = line["label"]
                    relations.extend([tuple(sorted(l)) for l in line["linking"]])
                    tokenized_inputs = self.tokenizer(
                        line["text"],
                        add_special_tokens=False,
                        return_offsets_mapping=True,
                        return_attention_mask=False,
                    )
                    text_length = 0
                    ocr_length = 0
                    bbox = []
                    last_box = None
                    for token_id, offset in zip(tokenized_inputs["input_ids"], tokenized_inputs["offset_mapping"]):
                        if token_id == 6:
                            bbox.append(None)
                            continue
                        text_length += offset[1] - offset[0]
                        tmp_box = []
                        while ocr_length < text_length:
                            ocr_word = line["words"].pop(0)
                            ocr_length += len(
                                self.tokenizer._tokenizer.normalizer.normalize_str(ocr_word["text"].strip())
                            )
                            tmp_box.append(simplify_bbox(ocr_word["box"]))
                        if len(tmp_box) == 0:
                            tmp_box = last_box
                        bbox.append(normalize_bbox(merge_bbox(tmp_box), size))
                        last_box = tmp_box
                    bbox = [
                        [bbox[i + 1][0], bbox[i + 1][1], bbox[i + 1][0], bbox[i + 1][1]] if b is None else b
                        for i, b in enumerate(bbox)
                    ]
                    mapper = {'B-QUESTION':1,'B-ANSWER':2,'B-HEADER':3,'I-QUESTION':4,'I-ANSWER':5,'I-HEADER':6}
                    if line["label"] == "other":
                        label = [0] * len(bbox)
                    else:
                        label = [mapper[f"I-{line['label'].upper()}"]] * len(bbox)
                        label[0] = mapper[f"B-{line['label'].upper()}"]
                    tokenized_inputs.update({"bbox": bbox, "labels": label})
                    if label[0] != 0:
                        entity_id_to_index_map[line["id"]] = len(entities)
                        entities.append(
                            {
                                "start": len(tokenized_doc["input_ids"]),
                                "end": len(tokenized_doc["input_ids"]) + len(tokenized_inputs["input_ids"]),
                                "label": line["label"].upper(),
                            }
                        )
                    for i in tokenized_doc:
                        tokenized_doc[i] = tokenized_doc[i] + tokenized_inputs[i]
                relations = list(set(relations))
                relations = [rel for rel in relations if rel[0] not in empty_entity and rel[1] not in empty_entity]
                kvrelations = []
                for rel in relations:
                    pair = [id2label[rel[0]], id2label[rel[1]]]
                    if pair == ["question", "answer"]:
                        kvrelations.append(
                            {"head": entity_id_to_index_map[rel[0]], "tail": entity_id_to_index_map[rel[1]]}
                        )
                    elif pair == ["answer", "question"]:
                        kvrelations.append(
                            {"head": entity_id_to_index_map[rel[1]], "tail": entity_id_to_index_map[rel[0]]}
                        )
                    else:
                        continue

                def get_relation_span(rel):
                    bound = []
                    for entity_index in [rel["head"], rel["tail"]]:
                        bound.append(entities[entity_index]["start"])
                        bound.append(entities[entity_index]["end"])
                    return min(bound), max(bound)

                relations = sorted(
                    [
                        {
                            "head": rel["head"],
                            "tail": rel["tail"],
                            "start_index": get_relation_span(rel)[0],
                            "end_index": get_relation_span(rel)[1],
                        }
                        for rel in kvrelations
                    ],
                    key=lambda x: x["head"],
                )

                chunk_size = 512
                for chunk_id, index in enumerate(range(0, len(tokenized_doc["input_ids"]), chunk_size)):
                    item = {}
                    for k in tokenized_doc:
                        item[k] = tokenized_doc[k][index : index + chunk_size]
                    entities_in_this_span = []
                    global_to_local_map = {}
                    for entity_id, entity in enumerate(entities):
                        if (
                            index <= entity["start"] < index + chunk_size
                            and index <= entity["end"] < index + chunk_size
                        ):
                            entity["start"] = entity["start"] - index
                            entity["end"] = entity["end"] - index
                            global_to_local_map[entity_id] = len(entities_in_this_span)
                            entities_in_this_span.append(entity)

                    regions_doc = regions[doc['id']+'.jpg']
                    region_id = {}
                    id_no = 0
                    for en in range(len(entities_in_this_span)):

                        entities_in_this_span[en]["bbox"] = merge_partial_boxes(entities_in_this_span[en], item)

                        bbox1 = item['bbox'][entities_in_this_span[en]['start']]
                        flag = False
                        temp_id = None
                        if entities_in_this_span[en]['start']+1 <entities_in_this_span[en]['end']:
                            bbox2 = item['bbox'][entities_in_this_span[en]['start']+1]
                            flag = True
                        bbox3 = item['bbox'][entities_in_this_span[en]['end']-1]

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
                            elif IoU(bbox1,reg) > 0 or IoU(bbox2,reg) > 0 or IoU(bbox3,reg) > 0:
                                if tuple(reg) in region_id:
                                    temp_id = region_id[tuple(reg)]
                                else:
                                    region_id[tuple(reg)] = id_no
                                    temp_id = id_no
                                    id_no += 1
                                break

                        if(temp_id == None):
                            reg = merge_partial_boxes(entities_in_this_span[en], item)

                            if tuple(reg) in region_id:
                                    temp_id = region_id[tuple(reg)]
                            else:
                                region_id[tuple(reg)] = id_no
                                temp_id = id_no
                                id_no += 1
                            
                            entities_in_this_span[en]['region_id'] = temp_id
                        else:
                            entities_in_this_span[en]['region_id'] = temp_id

                    regions_in_this_span = []
                    for tup in region_id.keys():
                        regions_in_this_span.append([region_id[tup],tup[0],tup[1],tup[2],tup[3]])

                    sorted(regions_in_this_span, key=lambda x: x[0])
                    
                    relations_in_this_span = []
                    for relation in relations:
                        if (
                            index <= relation["start_index"] < index + chunk_size
                            and index <= relation["end_index"] < index + chunk_size
                        ):
                            relations_in_this_span.append(
                                {
                                    "head": global_to_local_map[relation["head"]],
                                    "tail": global_to_local_map[relation["tail"]],
                                    "start_index": relation["start_index"] - index,
                                    "end_index": relation["end_index"] - index,
                                }
                            )

                    input_words = [self.tokenizer.decode(i) for i in item['input_ids']]
                    item.update(
                        {
                            "id": f"{doc['id']}_{chunk_id}",
                            "image": image,
                            "entities": entities_in_this_span,
                            "relations": relations_in_this_span,
                            "region_ids": regions_in_this_span,
                            "input_words": input_words
                        }
                    )
                    #final_dict[f"{doc['id']}_{chunk_id}"] = regions_in_this_span
                    yield f"{doc['id']}_{chunk_id}", item
