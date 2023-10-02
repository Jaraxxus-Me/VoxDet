from logging import debug
from mmcv import image
import torch, torchvision
print(torch.__version__, torch.cuda.is_available())

# Check MMDetection installation
import mmdet
print(mmdet.__version__)

# Check mmcv installation
from mmcv.ops import get_compiling_cuda_version, get_compiler_version
print(get_compiling_cuda_version())
print(get_compiler_version())

from mmdet.apis import inference_detector, init_detector, show_result_pyplot

import os
import sys
import cv2
import numpy as np

from mmdet.models import build_detector

import threading

# Choose to use a config and initialize the detector
CONFIG_NAME = 'oln_box_zid_cat.py'
CONFIG_PATH = os.path.join('/ws/ROS/src/zid3d_ros/Zid3D/configs/oln_box', CONFIG_NAME)

# Setup a checkpoint file to load
MODEL_NAME =  '1125/epoch_2.pth'
MODEL_PATH = os.path.join('/storage/models', MODEL_NAME)

# Setup p1 path
P1_obj =  '1237'
P1_PATH = os.path.join('/storage/real_data/P1', P1_obj)

if __name__=='__main__':
    model = init_detector(CONFIG_PATH, MODEL_PATH, device='cuda:0')
    model.init(p1_path=P1_PATH)
    # model.init()
    test_im = '/storage/real_data/P2/IMG_2955.jpg'
    im = cv2.imread(test_im)
    image = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    image_np = np.asarray(image)

    # Use the detector to do inference
    # NOTE: inference_detector() is able to receive both str and ndarray
    results = inference_detector(model, image_np)
