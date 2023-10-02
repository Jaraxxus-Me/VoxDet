import os.path as osp
import warnings
from collections import OrderedDict

import numpy as np
from torch.utils.data import Dataset
import torch

import torch
from glob import glob
from tqdm import tqdm
import torchvision.transforms as transforms
import cv2
import os
import json
import yaml
from .builder import DATASETS

IMG_EXT = ['png', 'jpg']

@DATASETS.register_module()
class ReconZidDataset(Dataset):
    def __init__(self, p1_path, video_num, ins_scale, input_size, test_mode=False,):
        print('This is Phase 1, Voxel Reconstruction Training Phase')
        self.imagefiles = image_dataloader(p1_path, ins_scale, video_num, test_mode)
        self.input_size = input_size
        self.test_mode = test_mode
        self.sz_path = os.path.join(p1_path, 'sz.npz')
        self.cam_path = os.path.join(p1_path, 'scene_gt_all.json')
        with open (self.cam_path, 'r') as f:
                self.cam = json.load(f)
        if os.path.isfile(self.sz_path) and os.path.isfile(self.cam_path):
            print('Loading object size from existing path (we will first crop then resize the image for recon)')
            self.obj_sz = np.load(self.sz_path, allow_pickle=True)['arr_0'].item()
        else:
            print('Start constructing object size')
            self.cam = {}
            self.obj_sz = {}
            video_paths = sorted(glob(p1_path+'/*/'))
            for v in tqdm(video_paths):
                obj_id = v.split('/')[-2]
                print('OBJ {}/{}'.format(obj_id, len(video_paths)))
                maskset = os.path.join(v, 'mask')
                self.obj_sz[obj_id] = cal_objsz(maskset)
            print('Constructing object camera and size done!')
            np.savez(self.sz_path, self.obj_sz)

        self.flag = np.ones(len(self), dtype=np.uint8)

    def __getitem__(self, index):
        # imgs
        imageset = self.imagefiles[index]
        obj_id = imageset[0].split('/')[-3]
        maskset = [f.replace('rgb', 'mask') for f in imageset]
        mask_images = [crop_image_loader(imageset[img_id], maskset[img_id], self.obj_sz[obj_id], self.input_size) for img_id in range(len(imageset))]
        images_rgb = masked_rgb_preprocess(mask_images)
        # poses
        imgids = [int(f.split('/')[-1][0:-4]) for f in imageset]
        cam_traj = self.cam2pose_rel(self.cam[str(obj_id)], imgids)
        data = {}
        data['rgb'] = images_rgb
        data['traj'] = cam_traj
        if self.test_mode:
            data['ids'] = imgids
        return data
            

    def __len__(self):
        return len(self.imagefiles)

    def cam2pose_rel(self, cam, ids):
        # relative to the first frame
        rela_r = torch.eye(3).unsqueeze(0).repeat(len(ids), 1, 1)
        rela_t = torch.zeros((len(ids), 3, 1))
        pre_r = torch.Tensor(cam[str(ids[0])][0]["cam_R_m2c"]).reshape(3, 3)
        for i, img_id in enumerate(ids):
            curr_r = torch.Tensor(cam[str(img_id)][0]["cam_R_m2c"]).reshape(3, 3)
            rela_r[i] = curr_r @ pre_r.T
            # rela_t[i, :] = curr_t - pre_t
        rela_pose_1 = torch.cat([rela_r, rela_t], dim=-1)
        # relative to the first frame, inverse
        rela_r = torch.eye(3).unsqueeze(0).repeat(len(ids), 1, 1)
        rela_t = torch.zeros((len(ids), 3, 1))
        pre_r = torch.Tensor(cam[str(ids[0])][0]["cam_R_m2c"]).reshape(3, 3)
        for i, img_id in enumerate(ids):
            curr_r = torch.Tensor(cam[str(img_id)][0]["cam_R_m2c"]).reshape(3, 3)
            rela_r[i] = (curr_r @ pre_r.T).T
            # rela_t[i, :] = curr_t - pre_t
        rela_pose_t = torch.cat([rela_r, rela_t], dim=-1)
        # both input
        rela_pose = torch.cat([rela_pose_1, rela_pose_t], dim=0)
        return rela_pose

def image_loader(path, input_size):
    image = cv2.imread(path)
    input_h, input_w = input_size, input_size
    image = cv2.resize(image, (input_w, input_h))
    return image

def cal_objsz(mask_path):
    mask_paths = sum([sorted(glob(mask_path+f'/*.{ext}')) for ext in IMG_EXT],[])
    obj_sz = 0
    for mask_path in mask_paths:
        mask = cv2.imread(mask_path)
        if (mask==0).all():
            return 4
        W, H = mask.shape[0], mask.shape[1]
        # mask to box:
        x, y = np.where(mask[:,:,0] != 0)
        obj_w = np.max(x) - np.min(x)
        obj_h = np.max(y) - np.min(y)
        local_obj_sz = np.max([obj_w, obj_h]) + 5
        local_obj_sz = np.min([local_obj_sz, W, H])
        if local_obj_sz>obj_sz:
            obj_sz = local_obj_sz
    return obj_sz

def crop_image_loader(rgb_path, mask_path, obj_sz, input_size):
    rgb = cv2.imread(rgb_path)
    mask = cv2.imread(mask_path)
    W, H = rgb.shape[0], rgb.shape[1]
    left_x = (W - obj_sz)//2
    left_y = (H - obj_sz)//2
    # crop obj center
    rgb = rgb[left_x:left_x+obj_sz, left_y:left_y+obj_sz, :]
    mask = mask[left_x:left_x+obj_sz, left_y:left_y+obj_sz, :]
    # resize
    input_h, input_w = input_size, input_size
    image = cv2.resize(rgb, (input_w, input_h))
    mask = cv2.resize(mask, (input_w, input_h))
    return image, mask

def rgb_preprocess(images):
    images = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in images]
    images = torch.stack([transforms.ToTensor()(image) for image in images])
    return images

def masked_rgb_preprocess(mask_images):
    images = [cv2.cvtColor(image[0], cv2.COLOR_BGR2RGB) for image in mask_images]
    masks = torch.stack([transforms.ToTensor()(image[1]) for image in mask_images])
    images = torch.stack([transforms.ToTensor()(image) for image in images])
    images = images*masks
    return images

def cam2pose_rel(cam, ids):
    # relative to the first frame
    rela_r = torch.eye(3).unsqueeze(0).repeat(len(ids), 1, 1)
    rela_t = torch.zeros((len(ids), 3, 1))
    pre_r = torch.Tensor(cam[str(ids[0])]["cam_R_w2c"]).reshape(3, 3)
    pre_t = torch.Tensor(cam[str(ids[0])]["cam_t_w2c"]).reshape(3, 1)
    for i, img_id in enumerate(ids):
        curr_r = torch.Tensor(cam[str(img_id)]["cam_R_w2c"]).reshape(3, 3)
        curr_t = torch.Tensor(cam[str(img_id)]["cam_t_w2c"]).reshape(3, 1)
        rela_r[i] = curr_r @ pre_r.T
        # rela_t[i, :] = curr_t - pre_t
    rela_pose_1 = torch.cat([rela_r, rela_t], dim=-1)
    # relative to the first frame, inverse
    rela_r = torch.eye(3).unsqueeze(0).repeat(len(ids), 1, 1)
    rela_t = torch.zeros((len(ids), 3, 1))
    pre_r = torch.Tensor(cam[str(ids[0])]["cam_R_w2c"]).reshape(3, 3)
    pre_t = torch.Tensor(cam[str(ids[0])]["cam_t_w2c"]).reshape(3, 1)
    for i, img_id in enumerate(ids):
        curr_r = torch.Tensor(cam[str(img_id)]["cam_R_w2c"]).reshape(3, 3)
        curr_t = torch.Tensor(cam[str(img_id)]["cam_t_w2c"]).reshape(3, 1)
        rela_r[i] = (curr_r @ pre_r.T).T
        # rela_t[i, :] = curr_t - pre_t
    rela_pose_t = torch.cat([rela_r, rela_t], dim=-1)
    # both input
    rela_pose = torch.cat([rela_pose_1, rela_pose_t], dim=0)
    return rela_pose

def cam2pose_rel_unseen(cam, ids, is_train):
    # relative to the first frame, all poses
    if not is_train:
        all_ids = list(range(160))
    else:
        all_ids = np.random.choice(40, 6, replace=True)
    rela_r = torch.eye(3).unsqueeze(0).repeat(len(all_ids), 1, 1)
    rela_t = torch.zeros((len(all_ids), 3, 1))
    pre_r = torch.Tensor(cam[str(ids[0])]["cam_R_w2c"]).reshape(3, 3)
    pre_t = torch.Tensor(cam[str(ids[0])]["cam_t_w2c"]).reshape(3, 1)
    for i, img_id in enumerate(all_ids):
        curr_r = torch.Tensor(cam[str(img_id)]["cam_R_w2c"]).reshape(3, 3)
        curr_t = torch.Tensor(cam[str(img_id)]["cam_t_w2c"]).reshape(3, 1)
        rela_r[i] = curr_r @ pre_r.T
        # rela_t[i, :] = curr_t - pre_t
    rela_pose_1 = torch.cat([rela_r, rela_t], dim=-1)
    # relative to the first frame, inverse, seen poses
    rela_r = torch.eye(3).unsqueeze(0).repeat(len(ids), 1, 1)
    rela_t = torch.zeros((len(ids), 3, 1))
    pre_r = torch.Tensor(cam[str(ids[0])]["cam_R_w2c"]).reshape(3, 3)
    pre_t = torch.Tensor(cam[str(ids[0])]["cam_t_w2c"]).reshape(3, 1)
    for i, img_id in enumerate(ids):
        curr_r = torch.Tensor(cam[str(img_id)]["cam_R_w2c"]).reshape(3, 3)
        curr_t = torch.Tensor(cam[str(img_id)]["cam_t_w2c"]).reshape(3, 1)
        rela_r[i] = (curr_r @ pre_r.T).T
        # rela_t[i, :] = curr_t - pre_t
    rela_pose_t = torch.cat([rela_r, rela_t], dim=-1)
    # both input
    rela_pose = torch.cat([rela_pose_1, rela_pose_t], dim=0)
    return rela_pose, all_ids

def image_dataloader(data_path, ids, video_num, recon_all=True):
    video_paths = sorted(glob(data_path+'/*/'))
    video_paths = [os.path.join(v_path, 'rgb') for v_path in video_paths]
    video_paths.sort()
    video_path = video_paths[ids[0]:ids[1]]
    batches = []
    print('Loading images...')
    while len(batches)<video_num:
        for index in tqdm(range(len(video_path))):
            video_batches = []
            vpath = video_path[index]
            fnames = sum([sorted(glob(vpath+f'/*.{ext}')) for ext in IMG_EXT],[])
            img_ids = list(range(0, len(fnames)))
            img_ids.sort()
            input_id = list(np.random.choice(img_ids, 32, replace=False))
            input_id.sort()
            recon_id = list(np.random.choice(img_ids, 4, replace=False))
            recon_id.sort()
            if recon_all:
                recon_id = img_ids
            input_id.extend(recon_id)
            frame_sequence = [fnames[img_id] for img_id in input_id]
            video_batches.append(frame_sequence)
            batches.extend(video_batches)
    print('Videos: {}'.format(len(batches)))
    return batches