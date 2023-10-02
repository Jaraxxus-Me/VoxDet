import os
from mmdet.datasets.pipelines import Compose
from tqdm import tqdm
import numpy as np

P1_base = 'data/BOP/ycbv/test_video_point'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# need to coment out the Collect function
P1_pipeline = [
    dict(type='LoadP1Info', target_sz=128),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImagesToTensor', keys=['rgb', 'mask', 'point']),
    dict(type='Collect', keys=['rgb', 'mask', 'point']),
]

pipeline = Compose(P1_pipeline)
obj_ids = os.listdir(P1_base)

for obj_id in tqdm(obj_ids):
    tar_path = os.path.join(P1_base, '{}'.format(obj_id))
    results = dict(P1_path=P1_base, obj_id=obj_id)
    data = pipeline(results)
    np.savez(os.path.join(tar_path, 'info.npz'), rgb=data['rgb'], mask=data['mask'], point=data['point'])
    # with open(os.path.join(tar_path, 'info.pkl'), 'wb') as f:
    #     pkl.dump(data, f)
