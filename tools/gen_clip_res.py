import torch
from torch import nn
import torch.nn.functional as F
import clip
import os
from tqdm import tqdm
from PIL import Image
import pickle as pkl
import argparse
from pycocotools.coco import COCO
import torchvision.transforms as T
import time
import glob
import numpy as np
import cv2

IMG_EXT = ['png', 'jpg']

def cal_objsz(mask_path):
    mask_paths = sum([sorted(glob.glob(mask_path+f'/*.{ext}')) for ext in IMG_EXT],[])
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
    x, y = np.where(mask[:,:,0] != 0)
    left_x = np.max([(np.max(x) + np.min(x))//2 - obj_sz//2, 0])
    left_y = np.max([(np.max(y) + np.min(y))//2 - obj_sz//2, 0])
    # crop obj center
    rgb = rgb[left_x:left_x+obj_sz, left_y:left_y+obj_sz, :]
    mask = mask[left_x:left_x+obj_sz, left_y:left_y+obj_sz, :]
    # resize
    input_h, input_w = input_size, input_size
    image = cv2.resize(rgb, (input_w, input_h))
    mask = cv2.resize(mask, (input_w, input_h))/255
    mask[mask<0.9] = 0
    return image

class PairwiseCosine(nn.Module):
    def __init__(self, inter_batch=False, dim=-1, eps=1e-8, T=0.07):
        super(PairwiseCosine, self).__init__()
        self.inter_batch, self.dim, self.eps, self.T = inter_batch, dim, eps, T
        self.eqn = 'amd,bnd->abmn' if inter_batch else 'bmd,bnd->bmn'

    def forward(self, x, y):
        x = F.normalize(x, dim=-1)
        y = F.normalize(y, dim=-1)
        xx = torch.sum(x**2, dim=self.dim).unsqueeze(-1) # (A, M, 1)
        yy = torch.sum(y**2, dim=self.dim).unsqueeze(-2) # (B, 1, N)
        if self.inter_batch:
            xx, yy = xx.unsqueeze(1), yy.unsqueeze(0) # (A, 1, M, 1), (1, B, 1, N)
        xy = torch.einsum(self.eqn, x, y) if x.shape[1] > 0 else torch.zeros_like(xx * yy)
        # return xy / (xx * yy).clamp(min=self.eps**2).sqrt()
        return xy

def parse_args():
    parser = argparse.ArgumentParser(
        description='Use CLIP to calculate cos_similarity for exhuastive 2D matching')
    parser.add_argument('--mode', default = 'max', help='max/mean mode for CLIP results')
    parser.add_argument('--dataset', default = 'ycbv', help='lmo/ycbv/RoboTools')
    parser.add_argument('--clip_model', default = 'ViT-B/32', help="['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']")
    parser.add_argument('--input', default = 'work_dirs/zid_rcnn_v1_base/bop_robo_12_cos.pkl', help='input result file in pickle format')
    parser.add_argument('--output', default = 'work_dirs/OLN_CLIP/bop_robo_ViT_L_14.pkl', help='input result file in pickle format')
    args = parser.parse_args()
    return args

def main():
    # prpare clip
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(args.clip_model, device=device)
    matcher = PairwiseCosine()

    # prepare dataset
    anno = os.path.join('data/BOP', args.dataset, 'test/scene_gt_coco_all.json')
    coco = COCO(anno)
    sz_path = os.path.join('data/BOP', args.dataset, 'test_video/sz.npz')

    # prepare sz for crop
    print('Starting to prepare temp feature dict:')
    if os.path.isfile(sz_path):
        print('Loading object size from existing path (we will first crop then resize the image for recon)')
        obj_sz = np.load(sz_path, allow_pickle=True)['arr_0'].item()
    else:
        print('Start constructing object size')
        obj_sz = {}
        video_paths = sorted(glob.glob(os.path.join('data/BOP', args.dataset, 'test_video')+'/*/'))
        for v in tqdm(video_paths):
            obj_id = int(v.split('/')[-2][-6:])
            print('OBJ {}/{}'.format(obj_id, len(video_paths)))
            maskset = os.path.join(v, 'mask')
            obj_sz[obj_id] = cal_objsz(maskset)
        print('Constructing object camera and size done!')
        np.savez(sz_path, obj_sz)

    temp_feat_dict = {}
    for cls_id in tqdm(coco.getCatIds()):
        # first calculate masked croped image and save
        tmp_info_folder = os.path.join('data/BOP', args.dataset, 'test_video/obj_{:06d}'.format(cls_id))
        tar_info_folder = os.path.join('data/BOP', args.dataset, 'test_video_pre/obj_{:06d}'.format(cls_id))
        os.makedirs(tar_info_folder, exist_ok=True)

        if not os.path.isfile(os.path.join(tar_info_folder, '000000.png')):
            print('First calculate masked croped image and save as png:')
            imageset = glob.glob(os.path.join(tmp_info_folder, 'rgb', "*.jpg"))
            imageset.sort()
            maskset = [f.replace('rgb', 'mask') for f in imageset]
            for img_id in range(len(imageset)):
                mask_image = crop_image_loader(imageset[img_id], maskset[img_id], obj_sz[cls_id], 512)
                tar_image_name = os.path.join(tar_info_folder, '{:06d}.png'.format(img_id))
                cv2.imwrite(tar_image_name, mask_image)
            print('Done for obj:{}'.format(cls_id))
        else:
            print('Obj:{} already done, start converting features'.format(cls_id))

        rgbs = glob.glob(os.path.join(tar_info_folder, "*.png"))
        rgbs.sort()
        M = len(rgbs)
        ref_img = []
        for i in range(0, M, 10):
            rgb = rgbs[i]
            image = preprocess(Image.open(rgb)).unsqueeze(0).to(device)
            ref_img.append(image)
        with torch.no_grad():
            image_features = model.encode_image(torch.cat(ref_img, dim=0))
        temp_feat_dict[cls_id] = image_features

    # load open-world res
    with open(args.input, 'rb') as f:
        open_pro = pkl.load(f)
    # img_ids
    print("Start Matching!")
    # new res
    new_res = []
    for img_id in tqdm(coco.getImgIds()):
        img_file = os.path.join('data/BOP', args.dataset, 'test', coco.imgs[img_id]['file_name'])
        cls_id = coco.imgToAnns[img_id][0]['category_id']
        temp_feat = temp_feat_dict[cls_id]
        img = Image.open(img_file)
        # get open world proposal features
        fragments = []
        local_res = np.zeros_like(open_pro[img_id-1][0])[:500]
        origin_boxes = open_pro[img_id-1][0][:, :4]
        for i, box in enumerate(origin_boxes[:500, :]):
            img_crop = img.crop((int(box[0]), int(box[1]), int(box[2]), int(box[3])))
            if 0 in img_crop.size:
                img_crop = img.crop((1, 1, 5, 5))
            processed_crop = preprocess(img_crop).unsqueeze(0)
            fragments.append(processed_crop)
            local_res[i, :4] = box
        fragments = torch.cat(fragments, dim=0).to(device)
        with torch.no_grad():
            query_features = model.encode_image(fragments)
        similarity = matcher(query_features.unsqueeze(0).repeat(1, 1, 1), temp_feat.unsqueeze(0)).squeeze()
        if args.mode == 'max':
            matching_score = similarity.max(dim=1).values
        else:
            matching_score = similarity.mean(dim=1)
        local_res[:, -1] = matching_score.cpu().numpy()
        new_res.append([local_res])
    print("Done!!")
    with open(args.output, 'wb') as f:
        pkl.dump(new_res, f)
        

if __name__ == '__main__':
    main()