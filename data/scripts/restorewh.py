# -*- coding: utf-8 -*-
"""
@Project : 
@FileName: 
@Author  :penghr 
@Time    :202x/xx/xx xx:xx
@Desc    : 恢复错误的宽高
"""
import os
import cv2
from xml.dom.minidom import parse

root = '/Users/hrpeng/Desktop/mask1544'

ANA = os.path.join(root,'Annotations')
IMG = os.path.join(root,'JPEGImages')


# DOMTree = parse('/Users/hrpeng/Desktop/mask/Annotations/0a2e79cf.xml')
# xml_root = DOMTree.documentElement
# size = xml_root.getElementsByTagName("size")[0]
# img_w = size.getElementsByTagName("width")[0].childNodes[0]
# print(img_w)
# img_w.nodeValue = str(999)
# img_h = size.getElementsByTagName("height")[0].childNodes[0].data
# with open('/Users/hrpeng/Desktop/mask/Annotations/0a2e79cf.xml', 'w') as f:
#     DOMTree.writexml(f)
# print(img_w)


for xml in os.listdir(ANA):
    try:
        jpg = xml.replace('.xml','.jpg')
        jpg_path = os.path.join(IMG,jpg)
        xml_path = os.path.join(ANA,xml)
        img = cv2.imread(jpg_path)
        img_h = img.shape[0]
        img_w = img.shape[1]
        DOMTree = parse(xml_path)
        xml_root = DOMTree.documentElement
        size = xml_root.getElementsByTagName("size")[0]
        w = size.getElementsByTagName("width")[0].childNodes[0]
        h = size.getElementsByTagName("height")[0].childNodes[0]
        w.nodeValue = img_w
        h.nodeValue = img_h
        with open(xml_path, 'w') as f:
            DOMTree.writexml(f)
    except Exception as e:
        print(e)

