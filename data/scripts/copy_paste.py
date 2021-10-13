# -*- coding: utf-8 -*-
"""
@Project : 
@FileName: 
@Author  :penghr 
@Time    :202x/xx/xx xx:xx
@Desc    :扣图生成无目标场景的训练集
"""
import random
import os
from xml.dom.minidom import parse
import numpy as np
import cv2

# img_path1 = '/Users/hrpeng/Desktop/截屏2021-07-22 16.58.23.png'
# img_path2 = '/Users/hrpeng/Desktop/7689_fire.jpg'
#
# im = cv2.imread(img_path1)
# obj = cv2.imread(img_path2)
#
# mask = 255 * np.ones(obj.shape, obj.dtype)
#
# # The location of the center of the src in the dst
# width, height, channels = im.shape
# center = (height//3 , width // 2)
#
# # Seamlessly clone src into dst and put the results in output
# normal_clone = cv2.seamlessClone(obj, im, mask, center, cv2.NORMAL_CLONE)
# # mixed_clone = cv2.seamlessClone(obj, im, mask, center, cv2.MIXED_CLONE)
#
# # # Write results
# # cv2.imwrite(folder + "normal_merge.jpg", normal_clone)
# # cv2.imwrite(folder + "fluid_merge.jpg", mixed_clone)
#
# cv2.imwrite('/Users/hrpeng/Desktop/1595821_normal_clone.jpg', normal_clone)
# # cv2.imwrite('/Users/hrpeng/Desktop/1595821_mixed_clone.jpg', mixed_clone)
# cv2.imshow('111',normal_clone)
# cv2.waitKey(0)

ori_img_dir = '/Users/hrpeng/Desktop/dubaochen_test' #图片根目录
obj_dir = '/Users/hrpeng/Desktop/chefhat' #切出来的目标路径
model_xml_path = '/Users/hrpeng/Desktop/workspace/yolov5/data/template.xml' #模板
if not os.path.exists(os.path.join(ori_img_dir, 'JPEGImages')):
    os.mkdir(os.path.join(ori_img_dir, 'JPEGImages'))
if not os.path.exists(os.path.join(ori_img_dir, 'Annotations')):
    os.mkdir(os.path.join(ori_img_dir, 'Annotations'))



for a in range(1):
    for i in os.listdir(ori_img_dir):
        print('+++++++++++++')
        ori_img_path = os.path.join(ori_img_dir,i)
        print('img_path:',ori_img_path)
        if ori_img_path.endswith('.jpg'):
            img = cv2.imread(ori_img_path)
            img_h = img.shape[0]
            img_w = img.shape[1]
            DOMTree = parse(model_xml_path)
            xml_root = DOMTree.documentElement
            size = xml_root.getElementsByTagName("size")[0]
            w = size.getElementsByTagName("width")[0].childNodes[0]
            h = size.getElementsByTagName("height")[0].childNodes[0]
            w.nodeValue = img_w
            h.nodeValue = img_h

            for j in random.sample(os.listdir(obj_dir), 4):
                obj_path = os.path.join(obj_dir,j)
                if obj_path.endswith('.jpg'):
                    obj = cv2.imread(obj_path)
                    obj_h = obj.shape[0]
                    obj_w = obj.shape[1]
                    obj_node = DOMTree.createElement('object')
                    xml_root.appendChild(obj_node)

                    obj_name_node = DOMTree.createElement('name')
                    obj_name_value = DOMTree.createTextNode("chefhat")
                    obj_name_node.appendChild(obj_name_value)
                    obj_node.appendChild(obj_name_node)

                    obj_pose_node = DOMTree.createElement('pose')
                    obj_pose_value = DOMTree.createTextNode("Unspecified")
                    obj_pose_node.appendChild(obj_pose_value)
                    obj_node.appendChild(obj_pose_node)

                    obj_truncated_node = DOMTree.createElement('truncated')
                    obj_truncated_value = DOMTree.createTextNode("1")
                    obj_truncated_node.appendChild(obj_truncated_value)
                    obj_node.appendChild(obj_truncated_node)

                    obj_difficult_node = DOMTree.createElement('difficult')
                    obj_difficult_value = DOMTree.createTextNode("0")
                    obj_difficult_node.appendChild(obj_difficult_value)
                    obj_node.appendChild(obj_difficult_node)

                    print('obj_path',obj_path)
                    mask = 255 * np.ones(obj.shape, obj.dtype)
                    width, height, channels = img.shape
                    center = (int(height / random.uniform(1.2,5.4)), int(width / random.uniform(1.2,5.4)))
                    obj_width, obj_height, obj_channels = obj.shape
                    # bbox
                    obj_bbox_node = DOMTree.createElement('bndbox')

                    obj_bbox_xmin_node = DOMTree.createElement('xmin')
                    obj_bbox_xmin_value = DOMTree.createTextNode(str(int(center[0]-obj_height/2)))
                    obj_bbox_xmin_node.appendChild(obj_bbox_xmin_value)
                    obj_bbox_node.appendChild(obj_bbox_xmin_node)

                    obj_bbox_ymin_node = DOMTree.createElement('ymin')
                    obj_bbox_ymin_value = DOMTree.createTextNode(str(int(center[1]-obj_width/2)))
                    obj_bbox_ymin_node.appendChild(obj_bbox_ymin_value)
                    obj_bbox_node.appendChild(obj_bbox_ymin_node)

                    obj_bbox_xmax_node = DOMTree.createElement('xmax')
                    obj_bbox_xmax_value = DOMTree.createTextNode(str(int(center[0]+obj_height/2)))
                    obj_bbox_xmax_node.appendChild(obj_bbox_xmax_value)
                    obj_bbox_node.appendChild(obj_bbox_xmax_node)

                    obj_bbox_ymax_node = DOMTree.createElement('ymax')
                    obj_bbox_ymax_value = DOMTree.createTextNode(str(int(center[1]+obj_width/2)))
                    obj_bbox_ymax_node.appendChild(obj_bbox_ymax_value)
                    obj_bbox_node.appendChild(obj_bbox_ymax_node)

                    obj_node.appendChild(obj_bbox_node)

                    img = cv2.seamlessClone(obj, img, mask, center, cv2.NORMAL_CLONE)
                    with open(os.path.join(ori_img_dir, 'Annotations', str(a) + '_' + i.replace('.jpg', '.xml')),
                              'w') as f:
                        DOMTree.writexml(f,addindent='	',newl='\n', encoding = "utf-8")
            print(os.path.join(ori_img_dir,'JPEGImages',str(a)+'_'+i))
            cv2.imwrite(os.path.join(ori_img_dir,'JPEGImages',str(a)+'_'+i), img)
