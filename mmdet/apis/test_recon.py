
import os.path as osp
import os
import torchvision.io as io

import mmcv
import torch
import torch.distributed as dist
from mmcv.image import tensor2imgs
from mmcv.runner import get_dist_info



def single_gpu_test_recon(model,
                    data_loader,
                    out_dir):
    model.eval()
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    obj_id = 0
    for i, data in enumerate(data_loader):
        input_ids = list(torch.cat(data['ids'][:32], dim=0).numpy())
        data.pop('ids')
        with torch.no_grad():
            result = model(return_loss=False, **data)
        input_clip = result['input_imgs'].squeeze(0)
        pred_clip = result['recon_img']
        target_clip = result['output_imgs'].squeeze(0)
        pred_clip = torch.clamp(pred_clip, 0, 1)
        ###### Output
        # add mark
        for i in range(target_clip.shape[0]):
            if i not in input_ids:
                pred_clip[i, :, :10, :10] = torch.tensor([0, 1, 0], device=pred_clip.device, dtype=pred_clip.dtype).unsqueeze(1).unsqueeze(1)
                target_clip[i, :, :10, :10] = torch.tensor([0, 1, 0], device=pred_clip.device, dtype=pred_clip.dtype).unsqueeze(1).unsqueeze(1)
        save_dir = os.path.join(out_dir, f"obj_{obj_id}")
        os.makedirs(save_dir, exist_ok=True)
        vid = (target_clip.permute(0,2,3,1) * 255).byte().cpu()
        io.write_video(save_dir+'/eval_video_true.mp4', vid, 1)
        pred = (pred_clip.permute(0,2,3,1) * 255).byte().cpu()
        io.write_video(save_dir+'/eval_video_pred.mp4', pred, 1)
        inp = (input_clip.permute(0,2,3,1) * 255).byte().cpu()
        io.write_video(save_dir+'/eval_video_input.mp4', inp, 1)
        obj_id += 1

        for _ in range(1):
            prog_bar.update()
    return None