import cv2
import numpy as np
from PIL import Image
import pdb
import os
import json
import argparse

argParser = argparse.ArgumentParser()
argParser.add_argument("--ip_dir", type=str, help="input directory with images")
argParser.add_argument("--op_dir", type=str, help="output directory")
args = argParser.parse_args()

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


def filter_bbox(boxes, shape):
    final_box = []
    areas = []
    for box in boxes:
        # thresh_h = 0.009 
        # thresh_w = 0.021
        # if box[2]<thresh_w*shape[1] or box[3]<thresh_h*shape[0]:
        #     continue
        # if box[3]<30:
        #     continue
        area = box[2]*box[3]
        # if area<10000:
        #     continue
        areas.append((area,box))
    areas.sort()
   
    for a in areas:
        flag = True
        for final in final_box:
            if a[1][0] < final[0] and a[1][1] < final[1]:
                # If bottom-right inner box corner is inside the bounding box
                if final[0] + final[2] < a[1][0] + a[1][2] and final[1] + final[3] < a[1][1] + a[1][3]:
                    flag = False
                    break
                else:
                    iou = IoU(a[1], final)
                    #print(a[1], final, iou)
                    if iou > 0 and iou <= 1:
                        flag = False
                        break
            
        if flag:
            final_box.append(a[1])

    return final_box

def sort_contours(cnts, method="left-to-right"):    # initialize the reverse flag and sort index
    reverse = False
    i = 0    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1    # construct the list of bounding boxes and sort them from top to
    # bottom
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes), key=lambda b:b[1][i], reverse=reverse))    # return the list of sorted contours and bounding boxes
    return (cnts, boundingBoxes)


def find_template(img):   #send image without inverting colors
    # Length(width) of kernel as 100th of total width
    kernel_len = np.array(img).shape[1]//100# Defining a vertical kernel to detect all vertical lines of image 
    ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_len))# Defining a horizontal kernel to detect all horizontal lines of image
    hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 1))# A kernel of 2x2
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

    #Use vertical kernel to detect and save the vertical lines in a jpg
    image_1 = cv2.erode(~img, ver_kernel, iterations=3)
    vertical_lines = cv2.dilate(image_1, ver_kernel, iterations=3)

    #Use horizontal kernel to detect and save the horizontal lines in a jpg
    image_2 = cv2.erode(~img, hor_kernel, iterations=3)
    horizontal_lines = cv2.dilate(image_2, hor_kernel, iterations=3)

    # Combine horizontal and vertical lines in a new third image, with both having same weight.
    img_vh = cv2.addWeighted(vertical_lines, 0.5, horizontal_lines, 0.5, 0.0)#Eroding and thesholding the image
    img_vh = cv2.erode(~img_vh, kernel, iterations=2)
    thresh, img_vh = cv2.threshold(img_vh,128,255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    return img_vh

path = args.ip_dir
export_dict = {}


for file_name in os.listdir(path):
    #read your file
    file=f'{path}/{file_name}'
    img = cv2.imread(file,0)
    color = cv2.imread(file)
    img.shape#thresholding the image to a binary image
    thresh,img_bin = cv2.threshold(img,128,255,cv2.THRESH_BINARY |cv2.THRESH_OTSU)#inverting the image 
    

    ###############comment for FUNSD #####################
    # img_vh = find_template(img_bin)

    # # Detect contours for following box detection
    # contours, hierarchy = cv2.findContours(img_vh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # # Sort all the contours by top to bottom.
    # contours, boundingBoxes = sort_contours(contours, method='top-to-bottom')

    # #Create list box to store all boxes in  
    box = []# Get position (x,y), width and height for every contour and show the contour on image

    # for c in contours:
    #     x, y, w, h = cv2.boundingRect(c)    
        
    #     if (w<img_vh.shape[1]*0.9 and h<img_vh.shape[0]*0.9):
    #         image = cv2.rectangle(img_vh,(x,y),(x+w,y+h),(0,255,0),8)
    #     i += 1
    ###########################################################

    img_vh = find_template(img_bin)

    contours, hierarchy = cv2.findContours(img_vh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)    
        
        if (w<img_vh.shape[1]*0.9 and h<img_vh.shape[0]*0.9):
            image = cv2.rectangle(img_vh,(x,y),(x+w,y+h),(0,255,0),8)
            box.append([x,y,w,h])

    temp = cv2.imread(file)
    final = filter_bbox(box, temp.shape)

    for_dict = []
    for ele in final:
        for_dict.append([ele[0],ele[1],ele[0]+ele[2],ele[1]+ele[3]])

    export_dict[file_name] = for_dict

    i = 0
    color = [(255,0,0),(0,255,0),(0,0,255)]
    for ele in final:
        temp = cv2.rectangle(temp, (ele[0],ele[1]), (ele[0]+ele[2], ele[1]+ele[3]),color[i],5)
        i += 1
        i = i % 3
    cv2.imwrite(f"{args.op_dir}/table/{file_name}", temp)

bbox_file = open(f"{args.op_dir}/table.json",'w')
json.dump(export_dict, bbox_file, indent = 6, ensure_ascii=False)