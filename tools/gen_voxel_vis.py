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

class GradCAM:
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def __call__(self, score, feature_map):
        self.model.zero_grad()
        score.backward(retain_graph=True)

        # Compute gradients of score w.r.t. feature_map
        gradients = torch.autograd.grad(score, feature_map, create_graph=True)[0]
        # gradients = gradients.squeeze()
        weights = torch.mean(gradients, dim=[3, 4, 5], keepdim=True)
        mask = torch.sum(weights * feature_map, dim=2)
        # ReLU on the heatmap to get only the features that have a positive influence on the class of interest
        mask = F.relu(mask)
        if mask.any():
            mask = mask / torch.max(mask)
        return mask.squeeze()
    
# Load model
cfg = Config.fromfile('configs/oln_box/zid_box_bop_3dgconv_mix.py')
model = build_detector(cfg.model).cuda()
model.eval()

checkpoint = load_checkpoint(model, 'work_dirs/VoxDet_p2_2/iter_60800.pth')

# Load dataset
cfg.data.test.test_mode = True
dataset = build_dataset(cfg.data.test)

# Create GradCAM object
cam_extractor = GradCAM(model=model)

# Process each image in the dataset
ress = {}
dataset_id = []

for cat_id in dataset.cat_ids:
    img_ids = dataset.coco.getImgIds(catIds=[cat_id])
    dataset_id.append(img_ids[0])

for j in tqdm(dataset_id):
    i = j-1
    # Get the image and label
    rgb = dataset[i]['rgb'].unsqueeze(0).cuda()
    # rgb = Variable(rgb.unsqueeze(0).cuda(), requires_grad=True)

    # Forward pass & compute CAM
    img = [dataset[i]['img'][0].unsqueeze(0).cuda()]
    img_metas = [[dataset[i]['img_metas'][0].data]]
    mask = dataset[i]['mask'].unsqueeze(0).cuda()
    traj = dataset[i]['traj'].unsqueeze(0).cuda()
    obj_id = dataset[i]['id']
    cam = False
    if obj_id in model.roi_head.p1_info.keys():
        continue
    with torch.no_grad():
        scores, res, inds, raw_score = model(img, img_metas,                      
                        rgb=rgb,
                        mask=mask,
                        traj=traj,
                        obj_id=obj_id,
                        rescale=True,
                        return_loss=False)

dataset_id = []
for i in tqdm(range(0, len(dataset))):
    # Get the image and label
    rgb = dataset[i]['rgb'].unsqueeze(0).cuda()
    # rgb = Variable(rgb.unsqueeze(0).cuda(), requires_grad=True)

    # Forward pass & compute CAM
    img = [Variable(dataset[i]['img'][0].unsqueeze(0).cuda(), requires_grad=True)]
    img_metas = [[dataset[i]['img_metas'][0].data]]
    mask = dataset[i]['mask'].unsqueeze(0).cuda()
    traj = dataset[i]['traj'].unsqueeze(0).cuda()
    obj_id = dataset[i]['id']
    assert obj_id in model.roi_head.p1_info.keys()
    sup = model.roi_head.p1_info[obj_id]['voxel_support']['3d_support']
    model.roi_head.p1_info[obj_id]['voxel_support']['3d_support'] = Variable(sup, requires_grad=True)
    scores, res, inds, raw_score = model(img, img_metas,                      
                     rgb=rgb,
                     mask=mask,
                     traj=traj,
                     obj_id=obj_id,
                     rescale=True,
                     return_loss=False)
    support_vox = model.roi_head.p1_info[obj_id]['voxel_support']['3d_support']
    anno = dataset.coco.anns[i+1]['bbox']
    s = res[0][0][:5, -1]
    box = res[0][0][:5, :4]
    iou = calculate_iou(box, np.array([anno[0], anno[1], anno[0]+anno[2], anno[1]+anno[3]]))
    if iou[0]>0.6:
        print(i)
        dataset_id.append(dataset_id)
        mask = cam_extractor(raw_score[0][inds[0]][0][0], support_vox)
        ress[i] = {
            'id': i,
            'obj': obj_id,
            'score': s, 
            'box': box, 
            'cams': mask.detach().cpu().numpy(),
            'iou': iou
        }
        with open('vox_vis/ycbv_vox_{}.pkl'.format(i), 'wb') as f:
            pkl.dump(ress, f)
with open('vox_vis/ycbv_dataid.pkl'.format(i), 'wb') as f:
    pkl.dump(dataset_id, f)
