import os
from mmdet.datasets.pipelines import Compose
from tqdm import tqdm
import numpy as np

P1_base = 'data/BOP/RoboTools/test_video'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# need to coment out the Collect function
P1_pipeline = [
    dict(type='LoadP1Info', target_sz=128),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImagesToTensor', keys=['rgb', 'mask']),
    dict(type='Collect', keys=['rgb', 'mask']),
]

pipeline = Compose(P1_pipeline)
obj_ids = [1] # ycbv
# [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21] # ycbv
# [1,5,6,8,9,10,11,12] # lmo

for obj_id in tqdm(obj_ids):
    tar_path = os.path.join(P1_base, 'obj_{:06d}'.format(obj_id))
    results = dict(P1_path=P1_base, obj_id='obj_{:06d}'.format(obj_id))
    data = pipeline(results)
    np.savez(os.path.join(tar_path, 'info.npz'), rgb=data['rgb'], mask=data['mask'])
    # with open(os.path.join(tar_path, 'info.pkl'), 'wb') as f:
    #     pkl.dump(data, f)
