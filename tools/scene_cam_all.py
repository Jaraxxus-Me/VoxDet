import numpy as np
import os
import json
from tqdm import tqdm

# path config
base_path = 'data/BOP/RoboTools/test_video'
seqs = os.listdir(base_path)
new_json = os.path.join(base_path, 'scene_gt_all.json')
content = {}

for seq in tqdm(seqs):
    seq_path = os.path.join(base_path, seq)
    if not os.path.isdir(seq_path):
        continue
    # print("start seq: {}".format(seq))
    seq_path = os.path.join(base_path, seq)
    cam_path_o = os.path.join(seq_path, 'scene_gt.json')

    # camera par
    with open (cam_path_o, 'r') as f:
        cam_para_o = json.load(f)

    content[seq] = cam_para_o

with open (new_json, 'w') as f:
    json.dump(content, f)
