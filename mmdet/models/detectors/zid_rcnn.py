from ..builder import DETECTORS
from .two_stage import TwoStageDetector
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import torchvision.transforms.functional as Ft

from mmdet.core.visualization import imshow_det_bboxes, imshow_det_multi_bboxes
import mmcv
from mmcv.runner import auto_fp16
import pickle as pkl

# visualization tool
# import cv2
# x = (img.permute(0,2,3,1).cpu().detach().numpy()[0]*255).astype(np.uint8)
# x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
# cv2.imwrite('p2_cv.png', x)

# x = (rgb.permute(0,1,3,4,2).cpu().detach().numpy()[0,0]*255).astype(np.uint8)
# x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
# cv2.imwrite('p1_cv.png', x)

# box = gt_bboxes[0].cpu().detach().numpy()[0]
# x = cv2.rectangle(x, (int(box[0]), int(box[1])), (int(box[2]),int(box[3])), [0, 0, 255], 3)



@DETECTORS.register_module()
class ZidRCNN(TwoStageDetector):
    """Implementation of `Zid R-CNN <https://arxiv.org/abs/1506.01497>`_"""

    def __init__(self,
                 neg_rpn,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 mode='det',
                 D=4,
                 neck=None,
                 pretrained=None):
        super(ZidRCNN, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)
        self.p1_2D = None
        self.neg_rpn = neg_rpn
        self.mode = mode
        self.D = D
        
    @property
    def with_rpn(self):
        """bool: whether the detector has RPN"""
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    @property
    def with_roi_head(self):
        """bool: whether the detector has a RoI head"""
        return hasattr(self, 'roi_head') and self.roi_head is not None

    def init_weights(self, pretrained=None):
        """Initialize the weights in detector.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        super(TwoStageDetector, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        if self.with_rpn:
            self.rpn_head.init_weights()
        if self.with_roi_head:
            self.roi_head.init_weights(pretrained)

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    @auto_fp16(apply_to=('img', ))
    def forward(self, img=None, img_metas=None, return_loss=True, **kwargs):
        """Calls either :func:`forward_train` or :func:`forward_test` depending
        on whether ``return_loss`` is ``True``.

        Note this setting will change the expected inputs. When
        ``return_loss=True``, img and img_meta are single-nested (i.e. Tensor
        and List[dict]), and when ``resturn_loss=False``, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.
        """
        if self.mode == 'recon':
            if return_loss:
                return self.forward_train_recon(**kwargs)
            else:
                return self.forward_test_recon(**kwargs)
        else:
            assert self.mode == 'det'
            if return_loss:
                return self.forward_train_det(img, img_metas, **kwargs)
            else:
                return self.forward_test_det(img, img_metas, **kwargs)

    def forward_train_det(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      rgb=None,
                      mask=None,
                      traj=None,
                      query_pose=None,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        B = img.shape[0]
        x = self.extract_feat(img)

        # P1 2D information
        # downsampling
        # print("Downsampling templets, rate {}".format(self.D))
        M = rgb.shape[1]
        rgb = rgb[:, range(0, M, self.D), :]
        mask = mask[:, range(0, M, self.D), :]
        traj = traj[:, range(0, M, self.D), :]
        # feat
        ref_rgb = self.extract_feat(rgb.flatten(0, 1))
        mask = mask[:,:,0].flatten(0,1)
        boxes_p1 = self.masks_to_boxes(mask)
        p1_2D = {'feat': ref_rgb, 'box': boxes_p1, 'mask': mask, 'traj': traj}
        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            # RPN only learn forground instance box
            if not self.neg_rpn:
                rpn_bboxes = [gt_box[0:1] for gt_box in gt_bboxes]
            else:
                rpn_bboxes = gt_bboxes
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x,
                img_metas,
                rpn_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
                                                 gt_bboxes, gt_labels, query_pose,
                                                 gt_bboxes_ignore, gt_masks, p1_2D,
                                                 **kwargs)
        losses.update(roi_losses)

        return losses

    def forward_train_recon(self,
                      rgb=None,
                      traj=None,
                      **kwargs):
        # P1 2D information
        # only extract input feat
        # M = rgb.shape[1]
        input_rgb = rgb[:, :32, :]
        # feat
        ref_rgb = self.extract_feat(input_rgb.flatten(0, 1))
        p1_2D = {'img': rgb, 'feat': ref_rgb, 'traj': traj}
        losses = dict()

        recon_losses = self.roi_head.forward_train_recon(p1_2D,
                                                 **kwargs)
        losses.update(recon_losses)

        return losses

    def forward_test_recon(self,
                      rgb=None,
                      traj=None,
                      **kwargs):
        # P1 2D information
        # downsampling
        # M = rgb.shape[1]
        input_rgb = rgb[:, :32, :]
        # feat
        ref_rgb = self.extract_feat(input_rgb.flatten(0, 1))
        p1_2D = {'img': rgb, 'feat': ref_rgb, 'traj': traj}

        output = self.roi_head.forward_test_recon(p1_2D,
                                                 **kwargs)

        return output
    
    def forward_test_det(self, imgs, img_metas, 
                     rgb=None,
                     mask=None,
                     traj=None,**kwargs):
        """
        Args:
            imgs (List[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (List[List[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch.
        """
        for var, name in [(imgs, 'imgs'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError(f'{name} must be a list, but got {type(var)}')

        num_augs = len(imgs)
        if num_augs != len(img_metas):
            raise ValueError(f'num of augmentations ({len(imgs)}) '
                             f'!= num of image meta ({len(img_metas)})')

        # NOTE the batched image size information may be useful, e.g.
        # in DETR, this is needed for the construction of masks, which is
        # then used for the transformer_head.
        for img, img_meta in zip(imgs, img_metas):
            img_meta[0]['batch_input_shape'] = tuple(img.size()[-2:])

        if num_augs == 1:
            # proposals (List[List[Tensor]]): the outer list indicates
            # test-time augs (multiscale, flip, etc.) and the inner list
            # indicates images in a batch.
            # The Tensor should have a shape Px4, where P is the number of
            # proposals.
            if 'proposals' in kwargs:
                kwargs['proposals'] = kwargs['proposals'][0]
            return self.simple_test(imgs[0], img_metas[0], rgb, mask, traj, **kwargs)
        else:
            assert imgs[0].size(0) == 1, 'aug test does not support ' \
                                         'inference with batch size ' \
                                         f'{imgs[0].size(0)}'
            # TODO: support test augmentation for predefined proposals
            assert 'proposals' not in kwargs
            return self.aug_test(imgs, img_metas, rgb, mask, traj, **kwargs)

    def init(self, p1_path=None, pkl_path='/ws/ROS/src/zid3d_ros/Zid3D/support_feat.pkl'):
        """Generate P1 support"""
        if not p1_path:
            with open(pkl_path, 'rb') as f:
                self.p1_2D = pkl.load(f)
            return

        p1_data = np.load(os.path.join(p1_path, 'info.npz'))
        rgb = torch.from_numpy(p1_data['rgb'].astype(np.float32)).unsqueeze(0).cuda()
        mask = torch.from_numpy(p1_data['mask'].astype(np.float32)).unsqueeze(0).cuda()
        # depth = torch.from_numpy(p1_data['depth'].astype(np.float32)).cuda()

        ref_rgb = self.extract_feat(rgb.flatten(0, 1))
        mask = mask[:,:,0].flatten(0,1)
        mask[mask>220] = 255
        mask[mask<210] = 0
        boxes_p1 = self.masks_to_boxes(mask)
        p1_2D = {'feat': ref_rgb, 'box': boxes_p1, 'mask': mask, 'traj': traj}
        with open(pkl_path, 'wb') as f:
            pkl.dump(self.p1_2D, f)

    def simple_test(self, img, img_metas,                      
                     rgb=None,
                     mask=None,
                     traj=None,
                     obj_id=None,
                     proposals=None, 
                     rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        B = img.shape[0]
        x = self.extract_feat(img)

        # P1 2D information
        if (rgb!=None):
            if not self.roi_head.save_p1:
                N = rgb.shape[1]
                # down sampling
                # print("Down sampling at: {}".format(self.D))
                sampling_id = list(range(0, N, self.D))
                rgb = rgb[:, sampling_id]
                mask = mask[:, sampling_id]
                traj = traj[:, sampling_id]
                ref_rgb = self.extract_feat(rgb.flatten(0, 1))
                mask = mask[:,:,0].flatten(0,1)
                boxes_p1 = self.masks_to_boxes(mask)
                p1_2D = {'feat': ref_rgb, 'box': boxes_p1, 'mask': mask, 'traj': traj}
            elif obj_id not in self.roi_head.p1_info.keys():
                N = rgb.shape[1]
                # down sampling
                # print("Down sampling at: {}".format(self.D))
                sampling_id = list(range(0, N, self.D))
                rgb = rgb[:, sampling_id]
                mask = mask[:, sampling_id]
                traj = traj[:, sampling_id]
                ref_rgb = self.extract_feat(rgb.flatten(0, 1))
                mask = mask[:,:,0].flatten(0,1)
                boxes_p1 = self.masks_to_boxes(mask)
                p1_2D = {'feat': ref_rgb, 'box': boxes_p1, 'mask': mask, 'traj': traj}
            else:
                p1_2D = None
        elif not rgb:
            assert self.p1_2D
            p1_2D = self.p1_2D

        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals
            
        return self.roi_head.simple_test(
            x, proposal_list, img_metas, p1_2D, obj_id, rescale=rescale)

    def aug_test(self, imgs, img_metas,               
                     rgb=None,
                     mask=None,
                     depth=None,proposals=None, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        B = imgs.shape[0]
        x = self.extract_feat(imgs)

        if self.support:
            ref_rgb = self.extract_feat(rgb.flatten(0, 1))
            x = self.support_guidance(x, ref_rgb, B)
        proposal_list = self.rpn_head.aug_test_rpn(x, img_metas)
        return self.roi_head.aug_test(
            x, proposal_list, img_metas, rescale=rescale)

    def support_guidance(self, query, support, support_mask, B):
        correlated_feat = []
        support_mask /= 255.0
        for l in range(len(support)):
            # BN, C, W, H
            C, W, H = support[l].shape[1], support[l].shape[2], support[l].shape[3]
            # B, N, C, W, H
            s_mask = Ft.resize(support_mask, (W, H)).view(B, -1, 1, W, H)
            s = support[l].view(B, -1, C, W, H)*s_mask
            # B, C, W, H
            s = s.mean(1, False)
            # B, C, 1, 1 -> 1, BC, 1, 1
            s = s.mean(dim=[2, 3], keepdim=True)
            s_kernel = s.flatten(0, 1).unsqueeze(0)
            # 1, BC, W, H
            q = query[l].flatten(0, 1).unsqueeze(0)
            # DW-Conv
            conv_feat = F.conv2d(q, s_kernel.permute(1,0,2,3), 
                groups=C*B) # attention map, 1, BC, W, H
            # B, C, W, H
            conv_feat = conv_feat.view(B, C, query[l].shape[2], query[l].shape[3])
            feat_dict = {'ori': query[l], 'rela': conv_feat}
            correlated_feat.append(feat_dict)
        return tuple(correlated_feat)

    def show_result(self,
                    img,
                    result,
                    label=0,
                    gt_box=None,
                    score_thr=0.05,
                    bbox_color=(0, 0, 255),
                    text_color=(0, 0, 255),
                    mask_color=None,
                    thickness=4,
                    font_scale=1.2,
                    font_size=20,
                    win_name='',
                    fig_size=(15, 10),
                    show=False,
                    wait_time=0,
                    out_file=None):
        """Draw `result` over `img`.

        Args:
            img (str or Tensor): The image to be displayed.
            result (Tensor or tuple): The results to draw over `img`
                bbox_result or (bbox_result, segm_result).
            score_thr (float, optional): Minimum score of bboxes to be shown.
                Default: 0.3.
            bbox_color (str or tuple(int) or :obj:`Color`):Color of bbox lines.
               The tuple of color should be in BGR order. Default: 'green'
            text_color (str or tuple(int) or :obj:`Color`):Color of texts.
               The tuple of color should be in BGR order. Default: 'green'
            mask_color (None or str or tuple(int) or :obj:`Color`):
               Color of masks. The tuple of color should be in BGR order.
               Default: None
            thickness (int): Thickness of lines. Default: 2
            font_scale (float): Font scales of texts. Default: 0.5
            font_size (int): Font size of texts. Default: 13
            win_name (str): The window name. Default: ''
            fig_size (tuple): Figure size of the pyplot figure.
                Default: (15, 10)
            wait_time (float): Value of waitKey param.
                Default: 0.
            show (bool): Whether to show the image.
                Default: False.
            out_file (str or None): The filename to write the image.
                Default: None.

        Returns:
            img (Tensor): Only if not `show` or `out_file`
        """
        img = mmcv.imread(img)
        img = img.copy()
        if isinstance(result, tuple):
            bbox_result, segm_result = result
            if isinstance(segm_result, tuple):
                segm_result = segm_result[0]  # ms rcnn
        else:
            bbox_result, segm_result = result, None
        bboxes = [np.vstack(bbox_result[i]) for i in range(len(bbox_result))]
        labels = [[
            np.full(1, label, dtype=np.int32)
            for i in enumerate(bbox_result[j])
        ]for j in range(len(bbox_result))]
        labels = [np.concatenate(labels[k]) for k in range(len(labels))]
        bbox_color=[(20, 20, 255), (255, 191, 0), (0, 165, 255)]
        text_color=[(20, 20, 255), (255, 191, 0), (0, 165, 255)]
        # if out_file specified, do not show image in window
        if out_file is not None:
            show = False
        # draw bounding boxes
        img = imshow_det_multi_bboxes(
            img,
            bboxes,
            labels,
            gt_box,
            None,
            score_thr=score_thr,
            bbox_color=bbox_color,
            text_color=text_color,
            mask_color=mask_color,
            thickness=thickness,
            font_scale=font_scale,
            font_size=font_size,
            win_name=win_name,
            fig_size=fig_size,
            show=show,
            wait_time=wait_time,
            out_file=out_file)

        if not (show or out_file):
            return img

    def masks_to_boxes(self, masks: torch.Tensor) -> torch.Tensor:
        """
        Compute the bounding boxes around the provided masks.

        Returns a [N, 4] tensor containing bounding boxes. The boxes are in ``(x1, y1, x2, y2)`` format with
        ``0 <= x1 < x2`` and ``0 <= y1 < y2``.

        Args:
            masks (Tensor[N, H, W]): masks to transform where N is the number of masks
                and (H, W) are the spatial dimensions.

        Returns:
            Tensor[N, 4]: bounding boxes
        """
        if masks.numel() == 0:
            return torch.zeros((0, 4), device=masks.device, dtype=torch.float)

        n, h, w = masks.shape

        bounding_boxes = torch.zeros((n, 4), device=masks.device, dtype=torch.float)

        for index, mask in enumerate(masks):
            if (mask==0).all():
                bounding_boxes[index, 0] = w//2 - 2
                bounding_boxes[index, 1] = h//2 - 2
                bounding_boxes[index, 2] = w//2 + 2
                bounding_boxes[index, 3] = h//2 + 2
            else:
                y, x = torch.where(mask != 0)
                bounding_boxes[index, 0] = torch.min(x)
                bounding_boxes[index, 1] = torch.min(y)
                bounding_boxes[index, 2] = torch.max(x)
                bounding_boxes[index, 3] = torch.max(y)

        return bounding_boxes
        