# -*- coding: utf-8 -*-
"""
@Project : 
@FileName: 
@Author  :penghr 
@Time    :202x/xx/xx xx:xx
@Desc    :融合所有数据集并生成数据集列表，更改task
"""
import os
import random
from shutil import copyfile, rmtree

import cv2
from tqdm import tqdm

task = 'fire'
DATASET_PATH = os.path.join('../../../dataset', task)
TARGET = os.path.join('../../../', task)
ANNO_TARGET = os.path.join(TARGET, 'Annotations')
IMG_TARGET = os.path.join(TARGET, 'JPEGImages')
# rmtree(TARGET)
os.makedirs(TARGET, exist_ok=True)
os.makedirs(ANNO_TARGET, exist_ok=True)
os.makedirs(IMG_TARGET, exist_ok=True)
dir_list = []

if task == 'fire':
    dir_list = ['fire2244', 'fire2056', 'fire1684', 'fire9274']
elif task == 'chefhat':  # 厨师帽
    dir_list = ['chefhat751', 'chefhat1119', 'chefhat1529', 'chefhat1543', 'chefhat1758', 'chefhat2026', 'chefhat2436',
                'chefhat2724', 'chefhat3244', 'chefhat3795', 'chefhat4260', 'chefhat5100']
elif task == 'mask':  # 口罩
    dir_list = ['mask204', 'mask224', 'mask319', 'mask386', 'mask422', 'mask1028', 'mask1237', 'mask1416', 'mask2238',
                'mask2392', 'mask3404']
elif task == 'smoking':  # 抽烟
    dir_list = ['smoking4130']
elif task == 'chefhatcloth':  # 厨师服
    dir_list = ['chefhatcloth481', 'chefhatcloth516', 'chefhatcloth1322', 'chefhatcloth1444', 'chefhatcloth1472', 'chefhatcloth1993', 'chefhatcloth2969']
elif task == 'test':  # 测试新增用
    dir_list = ['chefhatcloth']

print('=====开始合并=====')
for dir in dir_list:
    dataset_path = os.path.join(DATASET_PATH, dir)
    dataset_anno_path = os.path.join(dataset_path, 'Annotations')
    dataset_img_path = os.path.join(dataset_path, 'JPEGImages')
    ANNO_list = os.listdir(dataset_anno_path)
    ANNO_list.sort()
    # print(ANNO_list)
    with tqdm(total=len(ANNO_list)) as pbar:
        for file_name in ANNO_list:
            # print(file_name)
            # img = cv2.imread(os.path.join(dataset_img_path, file_name.split('.')[0] + '.jpg')) # 测试图片是否损坏
            # print(file_name)
            pbar.update(1)
            try:
                copyfile(os.path.join(dataset_img_path, file_name.split('.')[0] + '.jpg'),
                         os.path.join(IMG_TARGET, file_name.split('.')[0] + '.jpg'))
                copyfile(os.path.join(dataset_anno_path, file_name.split('.')[0] + '.xml'),
                         os.path.join(ANNO_TARGET, file_name.split('.')[0] + '.xml'))
            except Exception as e:
                print(e, file_name)
print('=====合并结束=====')

print('=====开始写入=====')
count = 0
os.makedirs(os.path.join(TARGET, 'ImageSets'), exist_ok=True)
os.makedirs(os.path.join(TARGET, 'ImageSets', 'Main'), exist_ok=True)
train_file = open(os.path.join(TARGET, 'ImageSets', 'Main', 'train.txt'), 'w')
val_file = open(os.path.join(TARGET, 'ImageSets', 'Main', 'val.txt'), 'w')
test_file = open(os.path.join(TARGET, 'ImageSets', 'Main', 'test.txt'), 'w')
for root, dirs, files in os.walk(ANNO_TARGET):  # 遍历统计
    for file in files:
        a = random.randint(0, 100)
        if file.endswith('xml'):
            file_name = os.path.splitext(file)[0]
            count += 1  # 统计文件夹下文件个数
            if a < 70:
                train_file.write("{}\n".format(file_name))
            elif 70 <= a & a < 90:
                val_file.write("{}\n".format(file_name))
            else:
                test_file.write("{}\n".format(file_name))
print('文件的总数量为：', count)
print('=====写入完毕=====')
