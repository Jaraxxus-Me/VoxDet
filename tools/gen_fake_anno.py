import json
import os
from tqdm import tqdm
from copy import copy
from glob import glob
import shutil

data_class = [{'id': i, 'name': str(i), 'supercategory': 'robotools'} for i in range(1, 21)]

base_path = 'data/BOP/RoboTools_special/extracted'
tar_path = 'data/BOP/RoboTools_special/extracted/test'

all_seqs = os.listdir(base_path)

for seq in tqdm(all_seqs):
    # init
    coco = {
    'images':[],
    'annotations':[],
    'categories': data_class,
    'info':{'description': 'RoboTools', 'url': '', 'version': '0.1.0', 'year': 2023, 'contributor': '', 'date_created': '2023-04-03 00:38:04.330294'},
    'license':[]
    }
    img_id = 0
    ori_seq_path = os.path.join(base_path, seq)
    if not os.path.isdir(ori_seq_path):
        continue
    ori_imgs = glob(os.path.join(ori_seq_path, '*.png'))
    tar_seq_path = os.path.join(tar_path, '{:s}'.format(seq), 'rgb')
    tar_json = os.path.join(tar_path, '{:s}'.format(seq), 'scene_gt_coco_ins.json')
    os.makedirs(tar_seq_path, exist_ok=True)
    coco_anno = copy(coco)

    for ori_img in ori_imgs:
        ori_anno = {}
        # fake anno
        box = [1, 1, 60, 60]
        if 'T' in seq:
            class_id = int(seq[1:3])
            ori_anno[class_id] = box
        else:
            for cls_i in range(1, 21):
                ori_anno[cls_i] = box
        # ori_anno_file = ori_img.replace('png', 'txt')
        # with open(ori_anno_file, 'r') as f:
        #     content = f.readlines()
        #     for line in content:
        #         c = line.split(',')
        #         # lx ty w h
        #         box = [int(c[0]), int(c[1]), int(c[2])-int(c[0]), int(c[3])-int(c[1])]
        #         class_id = int(c[-1])
        #         ori_anno[class_id] = box
        # copy img
        img_name = ori_img.split('/')[-1]
        new_path = os.path.join(tar_seq_path, img_name)
        shutil.copyfile(ori_img, new_path)
        for ann_id in ori_anno.keys():
            img_info_local = {
            'id': img_id, 
            'file_name': 'rgb/{}'.format(img_name), 
            'width': 1920, 'height': 1080, 
            'date_captured': '2022-10-11 00:38:04.343987', 
            'license': 1, 
            'coco_url': '', 
            'flickr_url': ''
            }
            ann_local = {
                    'id': img_id, 
                    'image_id': img_id, 
                    'category_id': ann_id, 
                    'iscrowd': 0, 
                    'area': ori_anno[ann_id][2]*ori_anno[ann_id][3], 
                    'bbox': ori_anno[ann_id], 
                    'size': [1080, 1920], 
                    'width': 1920, 'height': 1080, 
                    'ignore': False
                    }
            coco_anno['images'].append(img_info_local)
            coco_anno['annotations'].append(ann_local)
            img_id += 1
    with open(tar_json, 'w') as fj:
        json.dump(coco_anno, fj)