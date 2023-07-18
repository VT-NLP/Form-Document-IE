import os
import easyocr
import cv2
import numpy as np
import json

import argparse

argParser = argparse.ArgumentParser()
argParser.add_argument("--para", type=str, help="true/false")
argParser.add_argument("--ip_dir", type=str, help="input directory with images")
argParser.add_argument("--op_dir", type=str, help="output directory")
args = argParser.parse_args()

reader = easyocr.Reader(['en'], gpu =True)

bbox_dict = {}

def bbox_easyocr(img,text_threshold = 0.7, paragraph = False, slope_ths = 0.1, ycenter_ths = 0.5, height_ths = 0.5, width_ths = 0.5, x_ths = 1.0, y_ths = 0.5):
    box = []
    results = reader.readtext(img, text_threshold = text_threshold, paragraph = paragraph, slope_ths = slope_ths, ycenter_ths = ycenter_ths, height_ths = height_ths, width_ths = width_ths, x_ths = x_ths, y_ths = y_ths)
    pic = cv2.imread(img)
    for i in range(0, len(results)):
      # extract the bounding box coordinates of the text region from
      # the current result
      x1 = int(results[i][0][0][0])
      y1 = int(results[i][0][0][1])
      x2 = int(results[i][0][2][0])
      y2 = int(results[i][0][2][1])
      
      # extract the OCR text itself along with the confidence of the
      # text localization
      text = results[i][1]
      text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
      
      cv2.rectangle(pic, (x1, y1), (x2, y2), (0, 0, 255), 2)
      box.append([x1,y1,x2,y2])
    
    if args.para == 'true':
      cv2.imwrite(f"{args.op_dir}/easy_para/{img.split('/')[-1]}", pic)
    else:
      cv2.imwrite(f"{args.op_dir}/easy_line/{img.split('/')[-1]}", pic)
    bbox_dict[img.split('/')[-1]] = box

path = args.ip_dir

for file in os.listdir(path):
  if args.para == 'true':
     bbox_easyocr(path+'/'+file,text_threshold=0.7, paragraph= True, x_ths = 2, y_ths = 1)
     bbox_file = open(f"{args.op_dir}/easy_para.json",'w')
  else:
     bbox_easyocr(path+'/'+file,text_threshold=0.7, width_ths = 10, ycenter_ths=1)
     bbox_file = open(f"{args.op_dir}/easy_line.json",'w')

if args.para == 'true':
     bbox_file = open(f"{args.op_dir}/easy_para.json",'w')
else:
     bbox_file = open(f"{args.op_dir}/easy_line.json",'w')
  
json.dump(bbox_dict, bbox_file, indent = 6, ensure_ascii=False)