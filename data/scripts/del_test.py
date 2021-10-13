# -*- coding: utf-8 -*-
"""
@Project : 
@FileName: 
@Author  :penghr 
@Time    :202x/xx/xx xx:xx
@Desc    :删除标注不好的图片
"""

import os

root = '/Users/hrpeng/Desktop/chefhatcloth_test'
Annotations_path = os.path.join(root, 'labels')  # 原始标注路径
Images_path = os.path.join(root, 'images')  # 筛选完的带标注图像

print(len(os.listdir(Annotations_path)), len(os.listdir(Images_path)))

for i in os.listdir(Annotations_path):
    img_name = i.replace('.txt', '.jpg')
    if img_name not in os.listdir(Images_path):
        os.remove(os.path.join(Annotations_path, i))

print(len(os.listdir(Annotations_path)), len(os.listdir(Images_path)))
