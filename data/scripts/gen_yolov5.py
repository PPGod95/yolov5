# -*- coding: utf-8 -*-
"""
@Project :
@FileName:
@Author  :penghr
@Time    :202x/xx/xx xx:xx
@Desc    :根据列表生成训练集测试集，更改task
"""

import os
import shutil
from shutil import copyfile
from xml.dom.minidom import parse

import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

# 原始数据集
task = 'fire'
ORIGIN_ROOT = os.path.join('../../..', task)
IMAGE_SET_ROOT = os.path.join(ORIGIN_ROOT, 'ImageSets/Main')  # 图片区分文件的路径
IMAGE_PATH = os.path.join(ORIGIN_ROOT, 'JPEGImages')  # 图片的位置
ANNOTATIONS_PATH = os.path.join(ORIGIN_ROOT, 'Annotations')  # 数据集标签文件的位置
LABELS_ROOT = os.path.join(ORIGIN_ROOT, 'Labels')  # 进行归一化之后的标签位置
data_num = len(os.listdir(ANNOTATIONS_PATH))

# YOLO 需要的数据集形式的新数据集
# TARGET_PATH = '../..'
TARGET_IMAGES_PATH = os.path.join(ORIGIN_ROOT + str(data_num), 'images')  # 区分训练集、测试集、验证集的图片目标路径
TARGET_LABELS_PATH = os.path.join(ORIGIN_ROOT + str(data_num), 'labels')  # 区分训练集、测试集、验证集的标签文件目标路径


def cord_converter(size, box):
    """
    将标注的 xml 文件标注转换为 darknet 形的坐标
    :param size: 图片的尺寸： [w,h]
    :param box: anchor box 的坐标 [左上角x,左上角y,右下角x,右下角y,]
    :return: 转换后的 [x,y,w,h]
    """

    x1 = int(box[0])
    y1 = int(box[1])
    x2 = int(box[2])
    y2 = int(box[3])

    dw = np.float32(1. / int(size[0]))
    dh = np.float32(1. / int(size[1]))

    w = x2 - x1
    h = y2 - y1
    x = x1 + (w / 2)
    y = y1 + (h / 2)

    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return [x, y, w, h]


def save_label_file(img_jpg_file_name, size, img_box):
    """
    保存标签的解析文件
    :param img_jpg_file_name:
    :param size:
    :param img_box:
    :return:
    """
    global person
    global head
    global hat
    global fire
    global smoke
    global mouse
    global chefhat
    global chefclothes
    global with_mask
    global mask_weared_incorrect
    global without_mask
    global with_smoke
    global all_cls

    save_file_name = os.path.join(LABELS_ROOT, os.path.splitext(img_jpg_file_name)[0] + '.txt')

    with open(save_file_name, "a+") as f:

        for box in img_box:
            # print(box[0])
            all_cls.add(box[0])
            cls_num = -1
            if task == 'fire':  # 火焰
                if box[0] in ['fire', 'Fire']:
                    cls_num = 0
                    fire += 1
                elif box[0] in ['smoke', 'Smoke']:
                    cls_num = 1
                    smoke += 1

            elif task == 'chefhat':  # 厨师帽
                if box[0] == 'mouse':
                    cls_num = 0
                    mouse += 1
                elif box[0] == 'head':
                    cls_num = 1
                    head += 1
                elif box[0] == 'chefhat':
                    cls_num = 2
                    chefhat += 1
                else:
                    continue

            elif task == 'mask':  # 口罩
                # print(save_file_name,box[0])
                if box[0] in ['mask', 'face_mask', 'with_mask', 'mask-01', 'mask-02']:
                    cls_num = 0
                    with_mask += 1
                # elif box[0] == 'mask_weared_incorrect':
                #     cls_num = 1
                #     mask_weared_incorrect += 1
                elif box[0] in ['face', 'without_mask', 'mask_weared_incorrect', 'face-01', 'face-02']:
                    cls_num = 1
                    without_mask += 1
                else:
                    continue

            elif task == 'smoking':  # 抽烟
                if box[0] in ['face_mask', 'with_mask', 'mask-01', 'mask-02']:
                    cls_num = 0
                    with_mask += 1
                elif box[0] in ['face', 'without_mask', 'mask_weared_incorrect', 'face-01', 'face-02']:
                    cls_num = 1
                    without_mask += 1
                elif box[0] == 'with_smoke':
                    cls_num = 2
                    with_smoke += 1
                else:
                    continue

            elif task == 'chefhatcloth':  # 厨师服
                if box[0] == 'person':
                    cls_num = 0
                    person += 1
                elif box[0] in ['chefhat', "Chef's hat", 'blackhat']:
                    cls_num = 1
                    chefhat += 1
                elif box[0] in ['chefclothes', "Chef's clothes"]:
                    cls_num = 2
                    chefclothes += 1
                elif box[0] == 'head':
                    cls_num = 3
                    head += 1
                else:
                    continue
            else:
                continue
            new_box = cord_converter(size, box[1:])  # 转换坐标
            f.write(f"{cls_num} {new_box[0]} {new_box[1]} {new_box[2]} {new_box[3]}\n")


def test_dataset_box_feature(file_name, point_array):
    """
    使用样本数据测试数据集的建议框
    :param file_name: 图片文件名
    :param point_array: 全部的点 [建议框sx1,sy1,sx2,sy2]
    :return: None
    """
    print(file_name)
    im = Image.open(os.path.join(IMAGE_PATH, os.path.splitext(file_name)[0] + ".jpg"))
    im_draw = ImageDraw.Draw(im)
    for box in point_array:
        x1 = box[1]
        y1 = box[2]
        x2 = box[3]
        y2 = box[4]
        im_draw.rectangle((x1, y1, x2, y2), outline='red')
    im.show()


def get_xml_data(img_xml_file):
    """
    获取 xml 数据
    :param img_xml_file: 图片路径
    :return:
    """
    dom = parse(img_xml_file)
    xml_root = dom.documentElement
    # img_name = xml_root.getElementsByTagName("filename")[0].childNodes[0].data
    img_size = xml_root.getElementsByTagName("size")[0]
    objects = xml_root.getElementsByTagName("object")
    img_w = img_size.getElementsByTagName("width")[0].childNodes[0].data
    img_h = img_size.getElementsByTagName("height")[0].childNodes[0].data
    img_c = img_size.getElementsByTagName("depth")[0].childNodes[0].data
    img_box = []
    for box in objects:
        cls_name = box.getElementsByTagName("name")[0].childNodes[0].data
        x1 = int(box.getElementsByTagName("xmin")[0].childNodes[0].data)
        y1 = int(box.getElementsByTagName("ymin")[0].childNodes[0].data)
        x2 = int(box.getElementsByTagName("xmax")[0].childNodes[0].data)
        y2 = int(box.getElementsByTagName("ymax")[0].childNodes[0].data)
        img_box.append([cls_name, x1, y1, x2, y2])

    # test_dataset_box_feature(os.path.split(img_xml_file)[1], img_box) # 展示
    save_label_file(os.path.split(img_xml_file)[1], [img_w, img_h], img_box)


def copy_data(img_set_source, img_labels_root, imgs_source, dataset_type):
    """
    将标签文件和图片复制到最终数据集文件夹中
    :param img_set_source: 原数据集图片总路径
    :param img_labels_root: 生成的 txt 标签总路径
    :param imgs_source:
    :param dataset_type: 生成数据集的种类
    :return:
    """
    file_name = os.path.join(img_set_source, dataset_type + ".txt")  # 获取对应数据集种类的图片
    # 判断目标图片文件夹和标签文件夹是否存在，不存在则创建
    os.makedirs(os.path.join(TARGET_IMAGES_PATH, dataset_type), exist_ok=True)
    os.makedirs(os.path.join(TARGET_LABELS_PATH, dataset_type), exist_ok=True)
    with open(file_name, encoding="UTF-8") as f:
        for img_name in tqdm(f.read().splitlines()):
            try:
                img_sor_file = os.path.join(imgs_source, img_name + '.jpg')
                label_sor_file = os.path.join(img_labels_root, img_name + '.txt')

                # 复制图片
                dict_file = os.path.join(TARGET_IMAGES_PATH, dataset_type, img_name + '.jpg')
                copyfile(img_sor_file, dict_file)

                # 复制 label
                dict_file = os.path.join(TARGET_LABELS_PATH, dataset_type, img_name + '.txt')
                copyfile(label_sor_file, dict_file)
            except Exception as e:
                print(e, file_name)
                # img_sor_file = imgs_source.joinpath(img_name).with_suffix('.png')
                # label_sor_file = img_labels_root.joinpath(img_name).with_suffix('.txt')
                #
                # # 复制图片
                # dict_file = FILE_ROOT.joinpath(DEST_IMAGES_PATH, dataset_type, img_name).with_suffix('.png')
                # copyfile(img_sor_file, dict_file)
                #
                # # 复制 label
                # dict_file = FILE_ROOT.joinpath(DEST_LABELS_PATH, dataset_type, img_name).with_suffix('.txt')
                # copyfile(label_sor_file, dict_file)


if __name__ == '__main__':
    root = ANNOTATIONS_PATH  # 数据集 xml 标签的位置
    if os.path.exists(LABELS_ROOT):
        print("Cleaning Label dir for safety generating label, pls wait...")
        shutil.rmtree(LABELS_ROOT)  # 清空标签文件夹
        print("Cleaning Label dir done!")
    os.makedirs(LABELS_ROOT, exist_ok=True)  # 建立 Label 文件夹

    # 生成标签
    print("Generating Label files...")

    person = 0
    head = 0
    hat = 0
    fire = 0
    smoke = 0
    mouse = 0
    chefhat = 0
    with_mask = 0
    mask_weared_incorrect = 0
    without_mask = 0
    with_smoke = 0
    chefclothes = 0
    all_cls = set()

    with tqdm(total=len(os.listdir(root))) as p_bar:
        for file in os.listdir(root):
            p_bar.update(1)
            try:
                get_xml_data(os.path.join(root, file))
            except Exception as e:
                # os.remove(file)
                print(e, os.path.join(root, file))
    print('all_cls:', all_cls)
    print('person:', person)
    print('head:', head)
    print('hat:', hat)
    print('fire:', fire)
    print('smoke:', smoke)
    print('mouse:', mouse)
    print('chefhat:', chefhat)
    print('with_mask:', with_mask)
    print('mask_weared_incorrect:', mask_weared_incorrect)
    print('without_mask:', without_mask)
    print('with_smoke:', with_smoke)
    print('chefclothes:', chefclothes)

    # 将文件进行 train、val、test 的区分
    for dataset_input_type in ["train", "val", "test"]:
        print(f"Copying data {dataset_input_type}, pls wait...")
        copy_data(IMAGE_SET_ROOT, LABELS_ROOT, IMAGE_PATH, dataset_input_type)
    print('训练图片路径:', os.path.join('..', task + str(data_num), 'images', 'train'))

    # 创建空标签
    ROOT = '../../..'
    TARGET_DIR = task + str(data_num)
    DATASET_DIR = os.path.join(ROOT, 'dataset', 'noobject')
    IMG_TARGET_TRAIN = os.path.join(ROOT, TARGET_DIR, 'images', 'train')
    LABEL_TARGET_TRAIN = os.path.join(ROOT, TARGET_DIR, 'labels', 'train')
    print('目标路径:', IMG_TARGET_TRAIN)

    with tqdm(total=len(os.listdir(DATASET_DIR))) as pbar:
        for file_name in os.listdir(DATASET_DIR):
            try:
                copyfile(os.path.join(DATASET_DIR, file_name), os.path.join(IMG_TARGET_TRAIN, file_name))
                with open(os.path.join(LABEL_TARGET_TRAIN, file_name.replace('.jpg', '.txt')), 'w') as f:
                    pass
                pbar.update(1)
            except Exception as e:
                print(e, file_name)
