import os
import easyocr
import cv2
import numpy as np
import json


bbox_dict = {}
reader = easyocr.Reader(['en'], gpu =True)


def bbox_easyocr(img, folder_path ,text_threshold = 0.7, paragraph = False, slope_ths = 0.1, ycenter_ths = 0.5, height_ths = 0.5, width_ths = 0.5, x_ths = 1.0, y_ths = 0.5):
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
    
    cv2.imwrite(f"{folder_path}/{img.split('/')[-1]}", pic)

def generate_regions(folder_path):
  for file in os.listdir(os.path.join(folder_path,"images")):
    bbox_easyocr(os.path.join(folder_path,"images")+'/'+file, folder_path, text_threshold=0.7, paragraph= True, x_ths = 2, y_ths = 1)
    bbox_file = open(f"{folder_path}/regions.json",'w')  
    json.dump(bbox_dict, bbox_file, indent = 6, ensure_ascii=False)