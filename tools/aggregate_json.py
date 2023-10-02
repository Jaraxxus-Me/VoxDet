'''
Convert original BOP challenge json file to instance json file for all scenes
for instance json file, every annotation (on specific instance) is an individual image, the image ids is the same as anno ids
'''

import os
from re import L
from threading import local
from tqdm import tqdm
import json
import shutil
from pycocotools.coco import COCO
import numpy as np

dataset = 'RoboTools_special'
data_path = 'data/BOP'
base_anno = os.path.join(data_path, dataset, 'test/L01/scene_gt_coco_ins.json')
scenes = os.listdir(os.path.join(data_path, dataset, 'test'))

with open(base_anno, 'r') as f:
    anno_dict = json.load(f)

new_anno = anno_dict.copy()
new_anno["images"] = []
new_anno["annotations"] = []
new_im_id = 0
new_ann_id = 0

for scene in scenes:
    original_p2_anno = os.path.join(data_path, dataset, 'test/{:s}/scene_gt_coco_ins.json'.format(scene))
    coco = COCO(original_p2_anno)

    for im in tqdm(coco.imgs.keys()):
        loriginal_image_info = coco.imgs[im]
        anns = coco.loadAnns(ids=coco.getAnnIds(im))
        for ann in anns:
            local_img_info = loriginal_image_info.copy()
            ori_file = local_img_info['file_name']
            local_img_info['file_name'] = os.path.join(scene, ori_file)
            new_ann_id +=1
            new_im_id +=1
            ann['id'] = new_ann_id
            ann['image_id'] = new_im_id
            local_img_info['id'] = new_im_id
            new_anno["annotations"].append(ann)
            new_anno["images"].append(local_img_info)

with open(os.path.join(data_path, dataset, 'test/scene_gt_coco_all.json'), 'w') as f:
    json.dump(new_anno, f)

print('Total test images (split): {}'.format(new_im_id))
print('Total test instances (split): {}'.format(new_ann_id))