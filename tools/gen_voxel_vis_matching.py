import torch
import torch.nn.functional as F
import numpy as np
from mmcv.runner import load_checkpoint
from torch.autograd import Variable
from mmcv.runner import load_checkpoint
from mmcv import Config
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
import pickle as pkl
from tqdm import tqdm
from torchcam.methods import GradCAM
import argparse

def calculate_iou(pred_boxes, gt_box):
    # pred_boxes: a numpy array of shape (n, 4), where n is the number of boxes, 
    # and each box is defined by [xmin, ymin, xmax, ymax]
    # gt_box: a numpy array of shape (4, ), [xmin, ymin, xmax, ymax]

    # Calculate overlap in the x-direction
    x_overlap = np.maximum(0, np.minimum(pred_boxes[:, 2], gt_box[2]) - np.maximum(pred_boxes[:, 0], gt_box[0]))

    # Calculate overlap in the y-direction
    y_overlap = np.maximum(0, np.minimum(pred_boxes[:, 3], gt_box[3]) - np.maximum(pred_boxes[:, 1], gt_box[1]))

    # Calculate intersection
    intersection = x_overlap * y_overlap

    # Calculate areas of the predicted boxes and the ground truth box
    pred_areas = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
    gt_area = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])

    # Calculate union
    union = pred_areas + gt_area - intersection

    # Calculate IoU
    iou = intersection / union

    return iou

def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('--config', default = 'configs/oln_box/zid_box_bop_3dgconv_mix.py', help='test config file path')
    parser.add_argument('--checkpoint', default = 'work_dirs/VoxDet_p2_2/iter_60800.pth', help='checkpoint file')
    parser.add_argument('--img_id', default = 1, type=int, help='output result file in pickle format')
    parser.add_argument('--dataset', default = 'lmo', help='output result file in pickle format')
    args = parser.parse_args()

    return args

def main():
    # Load model
    args = parse_args()
    cfg = Config.fromfile(args.config)
    model = build_detector(cfg.model).cuda()
    checkpoint = load_checkpoint(model, args.checkpoint)
    model.eval()

    # Load dataset
    cfg.data.test.test_mode = True
    dataset = build_dataset(cfg.data.test)

    # Create GradCAM object
    target_layer = 'roi_head.relate_3d.relate3d'
    cam_extractor = GradCAM(model=model, target_layer=target_layer)

    # Process each image in the dataset
    ress = {}
    dataset_id = []
    i = args.img_id

    # Get the image and label
    rgb = dataset[i]['rgb'].unsqueeze(0).cuda()
    # rgb = Variable(rgb.unsqueeze(0).cuda(), requires_grad=True)

    # Forward pass & compute CAM
    img = [Variable(dataset[i]['img'][0].unsqueeze(0).cuda(), requires_grad=True)]
    img_metas = [[dataset[i]['img_metas'][0].data]]
    mask = dataset[i]['mask'].unsqueeze(0).cuda()
    traj = dataset[i]['traj'].unsqueeze(0).cuda()
    obj_id = dataset[i]['id']
    scores, res, inds, raw_score = model(img, img_metas,                      
                    rgb=rgb,
                    mask=mask,
                    traj=traj,
                    obj_id=obj_id,
                    rescale=True,
                    return_loss=False)
    anno = dataset.coco.anns[i+1]['bbox']
    s = res[0][0][:5, -1]
    box = res[0][0][:5, :4]
    iou = calculate_iou(box, np.array([anno[0], anno[1], anno[0]+anno[2], anno[1]+anno[3]]))
    dataset_id.append(dataset_id)
    mask = cam_extractor(list(raw_score[0].argmax(1)), raw_score[0])[0]
    mask_partial = mask[inds]
    ress[i] = {
        'id': i,
        'obj': obj_id,
        'score': s, 
        'box': box, 
        'cams': mask_partial.detach().cpu().numpy(),
        'iou': iou
    }
    with open('vox_vis/{}_vox_{}.pkl'.format(args.dataset, i), 'wb') as f:
        pkl.dump(ress, f)

if __name__ == '__main__':
    main()