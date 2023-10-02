import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import ConvModule
from mmcv.runner import force_fp32

from mmdet.core import multi_apply, multiclass_nms, build_bbox_coder
from mmdet.core.bbox import bbox_overlaps
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.losses import accuracy

from .bbox_head import BBoxHead
from .convfc_bbox_head import ConvFCBBoxHead

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

@HEADS.register_module()
class ConvFCBBoxSuperHead(ConvFCBBoxHead):
    r"""More general bbox scoring head, to construct the OLN-Box head. It
    consists of shared conv and fc layers and three separated branches as below.

    .. code-block:: none

                                    /-> cls fcs -> cls
        shared convs -> shared fcs
                                    \-> reg fcs -> reg

                                    \-> bbox-scoring fcs -> bbox-score
    """  # noqa: W605

    def __init__(self, 
                 neg_head=True,
                 with_bbox_score=True, 
                 contrastive=None, 
                 bbox_score_type='BoxIoU',
                 loss_bbox_score=dict(type='L1Loss', loss_weight=1.0),
                 loss_contra=dict(type='CrossEntropyLoss', loss_weight=1.0),
                 **kwargs):
        super(ConvFCBBoxSuperHead, self).__init__(**kwargs)
        self.with_bbox_score = with_bbox_score
        self.contra_topk = contrastive['test_topk']
        self.avg_conrta = contrastive['avg_conrta'] # average spatial for contrastive
        self.bg_conrta = contrastive['bg_conrta'] # do we use background for contrastive
        self.num_conrta = contrastive['num_conrta'] # how many maximu samples for contrastive?
        self.neg_head = neg_head
        self.cos_score = PairwiseCosine()
        if self.with_bbox_score:
            self.fc_bbox_score = nn.Linear(self.cls_last_dim, 1)

        self.loss_bbox_score = build_loss(loss_bbox_score)
        self.loss_contra = build_loss(loss_contra)
        self.contra_weights = self.loss_contra.loss_weight
        self.bbox_score_type = bbox_score_type

        self.with_class_score = self.loss_cls.loss_weight > 0.0
        self.with_bbox_loc_score = self.loss_bbox_score.loss_weight > 0.0

    def init_weights(self):
        super(ConvFCBBoxSuperHead, self).init_weights()
        if self.with_bbox_score:
            nn.init.normal_(self.fc_bbox_score.weight, 0, 0.01)
            nn.init.constant_(self.fc_bbox_score.bias, 0)

    def forward(self, x):
        # cos sim, use 2 features
        x_supp = x['support']
        B, N, C, W, H = x_supp.shape
        x_query = x['ori'].view(B, -1, C, W, H)
        # cls use relative feature
        x_cls = x['rela']
        # box and IoU regressor are open world, use original feature
        x_reg = x['ori']
        x_bbox_score = x['ori']
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x_cls = conv(x_cls)
                x_reg = conv(x_reg)
                x_bbox_score = conv(x_bbox_score)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
                x_reg = self.avg_pool(x_reg)
                x_bbox_score = self.avg_pool(x_bbox_score)

            x_cls = x_cls.flatten(1)
            x_reg = x_reg.flatten(1)
            x_bbox_score = x_bbox_score.flatten(1)

            for fc in self.shared_fcs:
                x_cls = self.relu(fc(x_cls))
                x_reg = self.relu(fc(x_reg))
                x_bbox_score = self.relu(fc(x_bbox_score))
        # separate branches
        cls_score = self.fc_cls(x_cls) if self.with_cls else None
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None
        bbox_score = (self.fc_bbox_score(x_bbox_score)
                      if self.with_bbox_score else None)
        
        if self.avg_conrta:
            x_query = x_query.mean((3, 4))
            x_supp = x_supp.mean((3, 4))
        contra_logits = self.cos_score(x_query.flatten(2), x_supp.flatten(2))

        contra_logits = contra_logits.view(-1, N)
        contra_logits = torch.max(contra_logits, dim=1, keepdim=True).values

        return cls_score, bbox_pred, bbox_score, contra_logits

    def _get_target_single(self, pos_bboxes, neg_bboxes, pos_gt_bboxes,
                           pos_gt_labels, cfg):
        num_pos = pos_bboxes.size(0)
        num_neg = neg_bboxes.size(0)
        num_samples = num_pos + num_neg

        # original implementation uses new_zeros since BG are set to be 0
        # now use empty & fill because BG cat_id = num_classes,
        # FG cat_id = [0, num_classes-1]
        labels = pos_bboxes.new_full((num_samples, ),
                                     self.num_classes,
                                     dtype=torch.long)
        label_weights = pos_bboxes.new_zeros(num_samples)
        bbox_targets = pos_bboxes.new_zeros(num_samples, 4)
        bbox_weights = pos_bboxes.new_zeros(num_samples, 4)
        bbox_score_targets = pos_bboxes.new_zeros(num_samples)
        bbox_score_weights = pos_bboxes.new_zeros(num_samples)

        if num_pos > 0:
            labels[:num_pos] = pos_gt_labels
            pos_weight = 1.0 if cfg.pos_weight <= 0 else cfg.pos_weight
            label_weights[:num_pos] = pos_weight

            if not self.reg_decoded_bbox:
                pos_bbox_targets = self.bbox_coder.encode(
                    pos_bboxes, pos_gt_bboxes)
            else:
                # When the regression loss (e.g. `IouLoss`, `GIouLoss`)
                # is applied directly on the decoded bounding boxes, both
                # the predicted boxes and regression targets should be with
                # absolute coordinate format.
                pos_bbox_targets = pos_gt_bboxes
            bbox_targets[:num_pos, :] = pos_bbox_targets
            bbox_weights[:num_pos, :] = 1
            
            # Bbox-IoU as target
            if self.bbox_score_type == 'BoxIoU':
                pos_bbox_score_targets = bbox_overlaps(
                    pos_bboxes, pos_gt_bboxes, is_aligned=True)
            # Centerness as target
            elif self.bbox_score_type == 'Centerness':
                tblr_bbox_coder = build_bbox_coder(
                    dict(type='TBLRBBoxCoder', normalizer=1.0))
                pos_center_bbox_targets = tblr_bbox_coder.encode(
                    pos_bboxes, pos_gt_bboxes)
                valid_targets = torch.min(pos_center_bbox_targets,-1)[0] > 0
                pos_center_bbox_targets[valid_targets==False,:] = 0
                top_bottom = pos_center_bbox_targets[:,0:2]
                left_right = pos_center_bbox_targets[:,2:4]
                pos_bbox_score_targets = torch.sqrt(
                    (torch.min(top_bottom, -1)[0] / 
                        (torch.max(top_bottom, -1)[0] + 1e-12)) *
                    (torch.min(left_right, -1)[0] / 
                        (torch.max(left_right, -1)[0] + 1e-12)))
            else:
                raise ValueError(
                    'bbox_score_type must be either "BoxIoU" (Default) or \
                    "Centerness".')

            bbox_score_targets[:num_pos] = pos_bbox_score_targets
            if self.neg_head:
                # IoU reg head also learn negtive boxes
                bbox_score_weights[:num_pos] = 1.0
            else:
                # IoU reg head doesn't learn negtive boxes
                bbox_score_weights[:num_pos] = 1 - pos_gt_labels

        if num_neg > 0:
            label_weights[-num_neg:] = 1.0
        
        # sample contrastive
        # label is always [0]
        contra_ind = pos_bboxes.new_zeros(num_samples)
        contra_labels = torch.zeros((1, 1), dtype=torch.long).cuda()
        contra_labels[0] = torch.tensor(0).cuda()
        # positive inds
        num_true_positive = torch.where(labels==0)[0].shape[0]
        contra_ind[torch.where(labels==0)[0][0]] = True
        # negtive inds
        contra_ind[torch.where(labels==1)[0][0:num_pos-num_true_positive]] = True
        if self.bg_conrta:
            curr_num = torch.where(contra_ind)[0].shape[0]
            if curr_num<self.num_conrta:
                contra_ind[torch.where(labels==1)[0][num_pos-num_true_positive:self.num_conrta-curr_num]] = True

        return (labels, label_weights, bbox_targets, bbox_weights,
                bbox_score_targets, bbox_score_weights, contra_labels, contra_ind)

    def get_targets(self,
                    sampling_results,
                    gt_bboxes,
                    gt_labels,
                    rcnn_train_cfg,
                    concat=True,
                    class_agnostic=False):
        pos_bboxes_list = [res.pos_bboxes for res in sampling_results]
        neg_bboxes_list = [res.neg_bboxes for res in sampling_results]
        pos_gt_bboxes_list = [res.pos_gt_bboxes for res in sampling_results]
        pos_gt_labels_list = [res.pos_gt_labels for res in sampling_results]
        (labels, label_weights, bbox_targets, bbox_weights, 
         bbox_score_targets, bbox_score_weights, contra_labels, contra_ind) = multi_apply(
            self._get_target_single,
            pos_bboxes_list,
            neg_bboxes_list,
            pos_gt_bboxes_list,
            pos_gt_labels_list,
            cfg=rcnn_train_cfg)

        if concat:
            labels = torch.cat(labels, 0)
            label_weights = torch.cat(label_weights, 0)
            bbox_targets = torch.cat(bbox_targets, 0)
            bbox_weights = torch.cat(bbox_weights, 0)
            bbox_score_targets = torch.cat(bbox_score_targets, 0)
            bbox_score_weights = torch.cat(bbox_score_weights, 0)
            # contra_labels = torch.cat(contra_labels, 0).squeeze(-1)
            # contra_ind = torch.cat(contra_ind, 0)

        return (labels, label_weights, bbox_targets, bbox_weights,
                bbox_score_targets, bbox_score_weights, contra_labels, contra_ind)

    @force_fp32(apply_to=('cls_score', 'bbox_pred', 'bbox_score', 'contra_logits'))
    def loss(self,
             cls_score,
             bbox_pred,
             bbox_score,
             contra_logits,
             rois,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             bbox_score_targets,
             bbox_score_weights,
             contra_labels,
             contra_ind,
             reduction_override=None):
        losses = dict()
        if cls_score is not None:
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
            if cls_score.numel() > 0:
                losses['loss_cls'] = self.loss_cls(
                    cls_score,
                    labels,
                    label_weights,
                    avg_factor=avg_factor,
                    reduction_override=reduction_override)
        if contra_logits is not None:
            B = len(contra_ind)
            losses['loss_contra'] = 0
            contra_logits = contra_logits.view(B, -1)
            tem_labels = labels.view(B, -1)
            for b in range(B):
                contra_logit = contra_logits[b]
                inds = contra_ind[b].to(bool)
                contra_label = contra_labels[b]

                local_label = tem_labels[b][inds]
                contra_logit = contra_logit[inds]
                assert torch.where(local_label==0)[0].shape[0] == 1
                contra_label[0] = torch.where(local_label==0)[0][0]
                contra_logit = contra_logit.view(contra_label.shape[0], -1)
                if contra_logits.numel() > 0:
                    losses['loss_contra'] += self.loss_contra(
                    contra_logit,
                    contra_label[0],
                    avg_factor=contra_logit.numel(),
                    reduction_override=reduction_override)
            losses['loss_contra'] /= B
            losses['loss_contra'] *= self.contra_weights

        if bbox_pred is not None:
            bg_class_ind = self.num_classes
            # 0~self.num_classes-1 are FG, self.num_classes is BG
            pos_inds = (labels >= 0) & (labels < bg_class_ind)
            # do not perform bounding box regression for BG anymore.
            if pos_inds.any():
                if self.reg_decoded_bbox:
                    # When the regression loss (e.g. `IouLoss`,
                    # `GIouLoss`, `DIouLoss`) is applied directly on
                    # the decoded bounding boxes, it decodes the
                    # already encoded coordinates to absolute format.
                    bbox_pred = self.bbox_coder.decode(rois[:, 1:], bbox_pred)
                if self.reg_class_agnostic:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), 4)[pos_inds.type(torch.bool)]
                else:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), -1,
                        4)[pos_inds.type(torch.bool),
                           labels[pos_inds.type(torch.bool)]]

                losses['loss_bbox'] = self.loss_bbox(
                    pos_bbox_pred,
                    bbox_targets[pos_inds.type(torch.bool)],
                    bbox_weights[pos_inds.type(torch.bool)],
                    avg_factor=bbox_targets.size(0),
                    reduction_override=reduction_override)
            else:
                losses['loss_bbox'] = bbox_pred[pos_inds].sum()

        if bbox_score is not None:
            if bbox_score.numel() > 0:
                losses['loss_bbox_score'] = self.loss_bbox_score(
                    bbox_score.squeeze(-1).sigmoid(),
                    bbox_score_targets,
                    bbox_score_weights,
                    avg_factor=bbox_score_targets.size(0),
                    reduction_override=reduction_override)

        # losses['acc'] = accuracy(cls_score[:,0:1].sigmoid()*bbox_score, labels)
        return losses

    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def get_bboxes(self,
                   rois,
                   cls_score,
                   bbox_pred,
                   bbox_score,
                   contra_logits,
                   rpn_score,
                   img_shape,
                   scale_factor,
                   rescale=False,
                   cfg=None):
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))
        # cls_score is not used.
        cls_scores = F.softmax(
            cls_score, dim=1)[:,0:1] if cls_score is not None else None
        # use max matching score for classification
        cls_scores = cls_scores.reshape(-1, bbox_pred.shape[0]).max(dim=0).values.unsqueeze(1)
        
        if bbox_pred is not None:
            bboxes = self.bbox_coder.decode(
                rois[:, 1:], bbox_pred, max_shape=img_shape)
        else:
            bboxes = rois[:, 1:].clone()
            if img_shape is not None:
                bboxes[:, [0, 2]].clamp_(min=0, max=img_shape[1])
                bboxes[:, [1, 3]].clamp_(min=0, max=img_shape[0])

        if rescale and bboxes.size(0) > 0:
            if isinstance(scale_factor, float):
                bboxes /= scale_factor
            else:
                scale_factor = bboxes.new_tensor(scale_factor)
                bboxes = (bboxes.view(bboxes.size(0), -1, 4) /
                          scale_factor).view(bboxes.size()[0], -1)

        scores = torch.sqrt(rpn_score * bbox_score.sigmoid() * cls_scores)
        _, inds = torch.sort(scores, dim=0, descending=True)
        scores = torch.cat([scores, torch.zeros_like(scores)], dim=-1)

        # cfg = None
        if cfg is None:
            return bboxes, scores
        else:
            det_bboxes, det_labels, inds = multiclass_nms(bboxes, 
                                                    scores,
                                                    cfg.score_thr, cfg.nms,
                                                    cfg.max_per_img, return_inds=True
                                                    )

            return det_bboxes, det_labels, inds


@HEADS.register_module()
class Shared2FCBBoxSuperHead(ConvFCBBoxSuperHead):

    def __init__(self, fc_out_channels=1024, *args, **kwargs):
        super(Shared2FCBBoxSuperHead, self).__init__(
            num_shared_convs=0,
            num_shared_fcs=2,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)
