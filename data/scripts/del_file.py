# -*- coding: utf-8 -*-
"""
@Project : 
@FileName: 
@Author  :penghr 
@Time    :202x/xx/xx xx:xx
@Desc    :删除标注不好的图片
"""

import os

root = '/Users/hrpeng/Desktop/chefhatcloth/chefhatcloth3447'
Annotations_path = os.path.join(root,'Annotations') # 原始标注路径
Images_path = os.path.join(root,'Images') # 筛选完的带标注图像
JPEGImages_path = os.path.join(root,'JPEGImages') # 原始图片路径

print(len(os.listdir(Annotations_path)),len(os.listdir(Images_path)),len(os.listdir(JPEGImages_path)))
diff_list= [x for x in os.listdir(JPEGImages_path) if x not in os.listdir(Images_path)]
for diff_file in diff_list:
    try:
        # print(os.path.join(JPEGImages_path,diff_file.replace('.jpg','.xml')))
        os.remove(os.path.join(JPEGImages_path,diff_file))
        os.remove(os.path.join(Annotations_path,diff_file.replace('.jpg','.xml')))
    except Exception as e:
        print(e,diff_file)

print(len(os.listdir(Annotations_path)),len(os.listdir(Images_path)),len(os.listdir(JPEGImages_path)))