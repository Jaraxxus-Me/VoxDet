import torch

from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler
from ..builder import HEADS, build_head
from .standard_roi_head import StandardRoIHead
import torch.nn.functional as F
import torch.nn as nn


@HEADS.register_module()
class ZidRoIHeadBase(StandardRoIHead):
    """OLN Box head.
    
    We take the top-scoring (e.g., well-centered) proposals from OLN-RPN and
    perform RoIAlign to extract the region features from each feature pyramid
    level. Then we linearize each region features and feed it through two fc
    layers, followed by two separate fc layers, one for bbox regression and the
    other for localization quality prediction. It is recommended to use IoU as
    the localization quality target in this stage. 
    """

    def __init__(self,
                 save_p1=False,
                 bbox_roi_extractor=None,
                 bbox_head=None,
                 mask_roi_extractor=None,
                 mask_head=None,
                 shared_head=None,
                 train_cfg=None,
                 test_cfg=None):
        super(ZidRoIHeadBase, self).__init__(
                 bbox_roi_extractor=bbox_roi_extractor,
                 bbox_head=bbox_head,
                 mask_roi_extractor=mask_roi_extractor,
                 mask_head=mask_head,
                 shared_head=shared_head,
                 train_cfg=train_cfg,
                 test_cfg=test_cfg)
        self.save_p1 = save_p1
        if self.save_p1:
            self.p1_info = {}

    def _bbox_forward(self, x, rois, p1_feats):
        """Box head forward function used in both training and testing."""
        # TODO: a more flexible way to decide which feature maps to use
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
        bbox_feat = {"ori": bbox_feats, 'support': p1_feats}
        cls_score, bbox_pred, bbox_score, contra_logits = self.bbox_head(bbox_feat)

        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats,
            bbox_score=bbox_score, contra_logits=contra_logits)
        return bbox_results

    def _bbox_forward_train(self, x, sampling_results, gt_bboxes, gt_labels,
                            img_metas, p1_2D):
        """Run forward function and calculate loss for box head in training."""
        B = x[0].shape[0]
        rois = bbox2roi([res.bboxes for res in sampling_results])
        p1_rois = bbox2roi([b.unsqueeze(0) for b in p1_2D['box']])
        p1_x = p1_2D['feat']
        p1_feats = self.bbox_roi_extractor(
            p1_x[:self.bbox_roi_extractor.num_inputs], p1_rois)
        p1_feats = p1_feats.reshape(B, -1, p1_feats.shape[1], p1_feats.shape[2], p1_feats.shape[3])
        bbox_results = self._bbox_forward(x, rois, p1_feats)

        bbox_targets = self.bbox_head.get_targets(sampling_results, gt_bboxes,
                                                  gt_labels, self.train_cfg)

        loss_bbox = self.bbox_head.loss(bbox_results['cls_score'],
                                        bbox_results['bbox_pred'], 
                                        bbox_results['bbox_score'],
                                        rois,
                                        *bbox_targets)

        bbox_results.update(loss_bbox=loss_bbox)
        return bbox_results

    def simple_test_bboxes(self,
                           x,
                           img_metas,
                           proposals,
                           rcnn_test_cfg,
                           p1_2D,
                           rescale=False):
        """Test only det bboxes without augmentation."""
        # RPN score
        B = x[0].shape[0]
        rpn_score = torch.cat([p[:, -1:] for p in proposals], 0)
        rois = bbox2roi(proposals)
        p1_rois = bbox2roi([b.unsqueeze(0) for b in p1_2D['box']])
        p1_x = p1_2D['feat']
        p1_feats = self.bbox_roi_extractor(
            p1_x[:self.bbox_roi_extractor.num_inputs], p1_rois)
        p1_feats = p1_feats.reshape(B, -1, p1_feats.shape[1], p1_feats.shape[2], p1_feats.shape[3])

        bbox_results = self._bbox_forward(x, rois, p1_feats)
        img_shapes = tuple(meta['img_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

        # split batch bbox prediction back to each image
        bbox_pred = bbox_results['bbox_pred']
        bbox_score = bbox_results['bbox_score']
        contra_logits = bbox_results['contra_logits']

        num_proposals_per_img = tuple(len(p) for p in proposals)
        rois = rois.split(num_proposals_per_img, 0)
        bbox_score = bbox_score.split(num_proposals_per_img, 0)          
        rpn_score = rpn_score.split(num_proposals_per_img, 0)
        contra_logits = contra_logits.split(num_proposals_per_img, 0) if contra_logits!=None else None

        # some detector with_reg is False, bbox_pred will be None
        if bbox_pred is not None:
            if isinstance(bbox_pred, torch.Tensor):
                bbox_pred = bbox_pred.split(num_proposals_per_img, 0)
            else:
                bbox_pred = self.bbox_head.bbox_pred_split(
                    bbox_pred, num_proposals_per_img)
        else:
            bbox_pred = (None, ) * len(proposals)

        # apply bbox post-processing to each image individually
        det_bboxes = []
        det_labels = []
        for i in range(len(proposals)):
            if contra_logits!=None:
                det_bbox, det_label = self.bbox_head.get_bboxes(
                    rois[i],
                    bbox_pred[i],
                    bbox_score[i],
                    contra_logits[i],
                    rpn_score[i],
                    img_shapes[i],
                    scale_factors[i],
                    rescale=rescale,
                    cfg=rcnn_test_cfg)
            else:
                det_bbox, det_label = self.bbox_head.get_bboxes(
                    rois[i],
                    bbox_pred[i],
                    bbox_score[i],
                    None,
                    rpn_score[i],
                    img_shapes[i],
                    scale_factors[i],
                    rescale=rescale,
                    cfg=rcnn_test_cfg)
            det_bboxes.append(det_bbox)
            det_labels.append(det_label)
        return det_bboxes, det_labels

    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      query_pose,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      p1_2D=None):
        """
        Args:
            x (list[Tensor]): list of multi-level img features.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            proposals (list[Tensors]): list of region proposals.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # assign gts and sample proposals
        if self.with_bbox or self.with_mask:
            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                assign_result = self.bbox_assigner.assign(
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                    gt_labels[i])
                sampling_result = self.bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(sampling_result)

        losses = dict()
        # bbox head forward and loss
        if self.with_bbox:
            bbox_results = self._bbox_forward_train(x, sampling_results,
                                                    gt_bboxes, gt_labels,
                                                    img_metas, p1_2D)
            losses.update(bbox_results['loss_bbox'])

        # mask head forward and loss
        if self.with_mask:
            mask_results = self._mask_forward_train(x, sampling_results,
                                                    bbox_results['bbox_feats'],
                                                    gt_masks, img_metas)
            losses.update(mask_results['loss_mask'])

        return losses

    def simple_test(self,
                    x,
                    proposal_list,
                    img_metas,
                    p1_2D,
                    proposals=None,
                    rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'

        det_bboxes, det_labels = self.simple_test_bboxes(
            x, img_metas, proposal_list, self.test_cfg, p1_2D, rescale=rescale)
        if torch.onnx.is_in_onnx_export():
            if self.with_mask:
                segm_results = self.simple_test_mask(
                    x, img_metas, det_bboxes, det_labels, rescale=rescale)
                return det_bboxes, det_labels, segm_results
            else:
                return det_bboxes, det_labels

        bbox_results = [
            bbox2result(det_bboxes[i], det_labels[i],
                        1)
            for i in range(len(det_bboxes))
        ]

        if not self.with_mask:
            return bbox_results
        else:
            segm_results = self.simple_test_mask(
                x, img_metas, det_bboxes, det_labels, rescale=rescale)
            return list(zip(bbox_results, segm_results))

    def support_guidance(self, query, support):
        # DW guidance
        B, C, W, H = support.shape[0],support.shape[2],support.shape[3],support.shape[4]
        # BN, C, W, H -> B, N, C, W, H
        q = query.view(B, -1, C, W, H)
        N = q.shape[1]
        s = support.mean(1, keepdim=True)
        # B, 1, C, W, H -> B, N, C, 1, 1
        s = s.repeat(1, N, 1, 1, 1).flatten(0,1)
        q = q.flatten(0,1)
        cat_feat = torch.cat([q, s], dim=1)
        rela_feat = self.support_guidance_block(cat_feat)
        return rela_feat
