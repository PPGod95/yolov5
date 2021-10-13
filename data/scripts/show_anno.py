# -*- coding: utf-8 -*-
"""
@Project : 
@FileName: 
@Author  :penghr 
@Time    :202x/xx/xx xx:xx
@Desc    :根据gt展示所有图片的标注
"""
import os
from xml.dom.minidom import parse

import cv2
from tqdm import tqdm

root = '/Users/hrpeng/Desktop/fire9274'  # voc格式根目录

ANA = os.path.join(root, 'Annotations')
IMG = os.path.join(root, 'JPEGImages')
if not os.path.exists(os.path.join(root, 'Images')):
    os.makedirs(os.path.join(root, 'Images'))

with tqdm(total=len(os.listdir(ANA))) as pbar:
    for xml in os.listdir(ANA):
        try:
            pbar.update(1)
            jpg = xml.replace('.xml', '.jpg')
            jpg_path = os.path.join(IMG, jpg)
            xml_path = os.path.join(ANA, xml)
            img = cv2.imread(jpg_path)
            DOMTree = parse(xml_path)
            xml_root = DOMTree.documentElement
            objs = xml_root.getElementsByTagName("object")
            for obj in objs:
                box = obj.getElementsByTagName("bndbox")[0]
                label = obj.getElementsByTagName("name")[0].childNodes[0].nodeValue
                x1 = int(box.getElementsByTagName("xmin")[0].childNodes[0].nodeValue)
                y1 = int(box.getElementsByTagName("ymin")[0].childNodes[0].nodeValue)
                x2 = int(box.getElementsByTagName("xmax")[0].childNodes[0].nodeValue)
                y2 = int(box.getElementsByTagName("ymax")[0].childNodes[0].nodeValue)
                if label in ['head','person']:
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(img, label, (x1, y1 - 2), 2, 1, (0, 0, 255))
                else:
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img, label, (x1, y1 - 2), 2, 1, (0, 255, 0))
            cv2.imwrite(os.path.join(root, 'Images', jpg), img)
        except Exception as e:
            # os.remove(os.path.join(ANA, xml))
            print(e,os.path.join(ANA, xml),os.path.join(IMG, jpg))

