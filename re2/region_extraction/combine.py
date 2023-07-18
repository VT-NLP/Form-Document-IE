import json
import cv2
from PIL import Image
import numpy as np
import easyocr
import pdb
import argparse
argParser = argparse.ArgumentParser()
argParser.add_argument("--lang", type=str, help="language")
argParser.add_argument("--ip_path", type=str, help="input path to images")
argParser.add_argument("--table", type=str, help="input path to table.json")
argParser.add_argument("--easy_para", type=str, help="input path to easy_para.json")
argParser.add_argument("--easy_line", type=str, help="input path to easy_line.json")
argParser.add_argument("--op_path", type=str, help="output path")

args = argParser.parse_args()


path = args.ip_path
reader = easyocr.Reader([args.lang,'en'], gpu =True)

with open(args.table) as json_file:
    table_dict = json.load(json_file)

with open(args.easy_para) as json_file:
    easy_dict_para = json.load(json_file)

with open(args.easy_line) as json_file:
    easy_dict_line = json.load(json_file)


def normalize_bbox(bbox, size):
    return [
        int(1000 * bbox[0] / size[0]),
        int(1000 * bbox[1] / size[1]),
        int(1000 * bbox[2] / size[0]),
        int(1000 * bbox[3] / size[1]),
    ]

def crop(file, table_box):
    im = Image.open(f'{path}/{file}')
    im = im.convert("L")
    final_table = []
    # i = 0
    for box in table_box:
        im_crop = im.crop((box[0],box[1],box[2],box[3]))
        crop_array = np.asarray(im_crop)
        equ = cv2.equalizeHist(crop_array)
        # Gaussian blur
        blur = cv2.GaussianBlur(equ, (5, 5), 1)

        # manual thresholding
        th2 = 60 # this threshold might vary!
        equ[equ>=th2] = 255
        equ[equ<th2]  = 0
        results = reader.readtext(crop_array)
        if (len(results) != 0):
            final_table.append(box)
    return final_table



def IoU(box1, box2):
    
    x1, y1, w1, h1 = box1
    x3, y3, w2, h2 = box2
    x2 = x1 + w1
    y2 = y1 + h1
    x4 = x3 + w2
    y4 = y3 + h2

    if (x1 < x3 and x2 < x3) or (x3 < x1 and x4 < x1):
        return 0
    
    if (y1 < y3 and y2 < y3) or (y3 < y1 and y4 < y1):
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

    

    area_1 = w1*h1
    area_2 = w2*h2

    area_union = area_1 + area_2 - area_inter

    iou = area_inter/area_union


    return iou


final_box_dict = {}

for file in table_dict.keys():
    table_box = table_dict[file]
    easy_para_box = easy_dict_para[file]
    easy_line_box = easy_dict_line[file]
    final_box_dict[file] = []
    final_box = crop(file, table_box)
    table_box = final_box
    
    final_para = []
    for box in easy_para_box:
        flag = True
        for b in table_box:
            iou = IoU(box,b)                
            if iou > 0.001 and iou <=1:
                flag = False
                break
        if flag:
            final_para.append(box)
    #pdb.set_trace()

    final_box = final_box + final_para 
    
    final_line = []
    for box in easy_line_box:
        flag = True
        for b in final_box:
            iou = IoU(box,b)
            if iou > 0.001 and iou <= 1:
                flag = False
                break
        if flag:
            final_line.append(box)

    final_box = final_box + final_line

    temp = cv2.imread(path+'/'+file)
    i = 0
    color = [(255,0,0),(0,255,0),(0,0,255)]
    for ele in final_box:
        new_box = [ele[0], ele[1], ele[2], ele[3]]
        final_box_dict[file].append(new_box)
        temp = cv2.rectangle(temp, (ele[0],ele[1]), (ele[2],ele[3]),color[i],5)
        i += 1
        i = i % 3
    cv2.imwrite(f"{args.op_path}/{file}", temp)

final_file = open(f"{args.op_path}/xfun_{args.lang}.json",'w')
json.dump(final_box_dict, final_file, indent = 6, ensure_ascii=False)

        
