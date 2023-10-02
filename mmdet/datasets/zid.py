from copy import copy
import itertools
import json
import logging
import os.path as osp
import tempfile
from collections import OrderedDict
import os
import mmcv
import numpy as np
import torch
from mmcv.utils import print_log
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from terminaltables import AsciiTable
import pickle as pkl
import time
from tqdm import tqdm

from mmdet.core import eval_recalls
from .builder import DATASETS
from .custom import CustomDataset

try:
    import pycocotools
    if not hasattr(pycocotools, '__sphinx_mock__'):  # for doc generation
        assert pycocotools.__version__ >= '12.0.2'
except AssertionError:
    raise AssertionError('Incompatible version of pycocotools is installed. '
                         'Run pip uninstall pycocotools first. Then run pip '
                         'install mmpycocotools to install open-mmlab forked '
                         'pycocotools.')


@DATASETS.register_module()
class ZidDataset(CustomDataset):
    def __init__(self, 
                 p1_path,
                 img_scale=-1,
                 ins_scale=-1,
                 **kwargs):
        # We convert all category IDs into 1 for the class-agnostic training and
        # evaluation. We train on train_class and evaluate on eval_class split.
        self.p1_path = p1_path
        with open(os.path.join(self.p1_path, 'scene_gt_all.json'), 'r') as f:
            self.cam_pos = json.load(f)
        with open(os.path.join(self.p1_path, 'ZID-10k_mapper_new.json'), 'r') as f:
            self.all_obj_dict = json.load(f)
        self.obj_ids = list(self.all_obj_dict.values())
        self.class_names_dict = ['ins_{:06d}'.format(i) for i in self.obj_ids]
        self.img_scale = img_scale
        self.ins_scale = ins_scale

        super(ZidDataset, self).__init__(**kwargs)

    def load_annotations(self, ann_file):
        """Load annotation from COCO style annotation file.

        Args:
            ann_file (str): Path of annotation file.

        Returns:
            list[dict]: Annotation info from COCO api.
        """

        self.coco = COCO(ann_file)
        self.cat_ids = self.coco.get_cat_ids(cat_names=self.class_names_dict)
        self.cat2label = {cat_id: 0 for cat_id in self.cat_ids}
        # self.img_ids = self.coco.get_img_ids()
        # np.random.shuffle(self.img_ids)
        # ins_dict = {}
        # for img_id in self.img_ids:
        #     if self.coco.load_anns(self.coco.getAnnIds(img_id)) == []:
        #         print(img_id)
        #         continue
        #     num_ins = self.coco.load_anns(self.coco.getAnnIds(img_id))[0]['num_ins']
        #     ins_dict[img_id] = num_ins
        # new_ins_dict = dict(sorted(ins_dict.items(), key=lambda item: item[1], reverse=True))
        # new_ids = list(new_ins_dict.keys())
        # np.save(ann_file[:-5]+"_0.npy", new_ids)
        self.img_ids = np.load(ann_file[:-5]+"_0.npy")
        # np.random.shuffle(self.img_ids)
        self.img_ids = self.img_ids[:self.img_scale]
        data_infos = []
        for i in self.img_ids:
            info = self.coco.load_imgs([i])[0]
            info['filename'] = info['file_name']
            anns = self.coco.getAnnIds(info['id'])
            for i, ann in enumerate(self.coco.load_anns(anns)):
                local_info = copy(info)
                local_info['ann_id'] = i
                local_info['obj_id'] = ann['category_id']
                assert local_info['obj_id'] in self.obj_ids, 'Anno_{} has object that do not have P1 video'.format(ann['id'])
                data_infos.append(local_info)
        np.random.shuffle(data_infos)
        data_infos = data_infos[:self.ins_scale]
        print('Dataset scale (before filtering):\n Images:' + str(len(self.img_ids)) + '\n Instances:' +
                str(len(data_infos)))
        return data_infos

    def get_ann_info(self, idx):
        """Get COCO annotation by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """
        img_id = self.data_infos[idx]['id']
        ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
        ann_info = self.coco.load_anns(ann_ids)
        return self._parse_ann_info(self.data_infos[idx], [ann_info[self.data_infos[idx]['ann_id']]])

    def get_cat_ids(self, idx):
        """Get COCO category ids by index.

        Args:
            idx (int): Index of data.

        Returns:
            list[int]: All categories in the image of specified index.
        """

        img_id = self.data_infos[idx]['id']
        ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
        ann_info = self.coco.load_anns(ann_ids)
        return [ann['category_id'] for ann in ann_info]

    def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths."""
        valid_inds = []
        valid_img_ids = []
        # obtain images that contain annotation
        for i, img_info in enumerate(self.data_infos):
            img_id = self.data_infos[i]['id']
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
                valid_img_ids.append(img_id)
        self.img_ids = valid_img_ids
        return valid_inds

    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys \
                introduced by pipeline.
        """
        img_info = self.data_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        data = self.pipeline(results)
        p1_data = np.load(os.path.join(self.p1_path, str(img_info['obj_id']), 'info.npz'))
        data['rgb'] = torch.from_numpy(p1_data['rgb'].astype(np.float32))
        data['mask'] = torch.from_numpy(p1_data['mask'].astype(np.float32))
        imgids = list(range(40))
        cam_traj = self.cam2pose_rel(self.cam_pos[str(img_info['obj_id'])], imgids)
        cam_traj_q = self.cam2pose_rel_q(self.cam_pos[str(img_info['obj_id'])], ann_info['cam_R_m2c'])
        data['traj'] = cam_traj
        data['query_pose'] = cam_traj_q
        return data
    
    def prepare_test_img(self, idx):
        """Get testing data  after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Testing data after pipeline with new keys intorduced by \
                piepline.
        """

        img_info = self.data_infos[idx]
        results = dict(img_info=img_info)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        data = self.pipeline(results)
        p1_data = np.load(os.path.join(self.p1_path, str(img_info['obj_id']), 'info.npz'))
        data['rgb'] = torch.from_numpy(p1_data['rgb'].astype(np.float32))
        data['mask'] = torch.from_numpy(p1_data['mask'].astype(np.float32))
        # data['depth'] = torch.from_numpy(p1_data['depth'].astype(np.float32))
        imgids = list(range(40))
        cam_traj = self.cam2pose_rel(self.cam_pos[str(img_info['obj_id'])], imgids)
        data['traj'] = cam_traj
        data['id'] = img_info['obj_id']
        return data

    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,\
                labels, masks, seg_map. "masks" are raw annotations and not \
                decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []
        cam_R_m2c = []
        for i, ann in enumerate(ann_info):
            assert len(ann_info) == 1
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            obj_w = np.array(ann['obj_w']).reshape(4, 4)[:3, :3]
            cam_w = np.array(ann['cam_w']).reshape(4, 4)[:3, :3]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                cam_R_m2c.append(cam_w @ obj_w.T)
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
                gt_masks_ann.append(ann.get('segmentation', None))

                for neg_box in ann['neg_box']:
                    x1, y1, w, h = neg_box
                    inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
                    inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
                    if inter_w * inter_h == 0:
                        continue
                    if ann['area'] <= 0 or w < 1 or h < 1:
                        continue
                    neg_bbox = [x1, y1, x1 + w, y1 + h]
                    gt_bboxes.append(neg_bbox)
                    gt_labels.append(1)

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        seg_map = img_info['filename'].replace('jpg', 'png')

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            cam_R_m2c=cam_R_m2c,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            seg_map=seg_map)

        return ann

    def xyxy2xywh(self, bbox):
        """Convert ``xyxy`` style bounding boxes to ``xywh`` style for COCO
        evaluation.

        Args:
            bbox (numpy.ndarray): The bounding boxes, shape (4, ), in
                ``xyxy`` order.

        Returns:
            list[float]: The converted bounding boxes, in ``xywh`` order.
        """

        _bbox = bbox.tolist()
        return [
            _bbox[0],
            _bbox[1],
            _bbox[2] - _bbox[0],
            _bbox[3] - _bbox[1],
        ]

    def _proposal2json(self, results):
        """Convert proposal results to COCO json style."""
        json_results = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            bboxes = results[idx]
            for i in range(bboxes.shape[0]):
                data = dict()
                data['image_id'] = img_id
                data['bbox'] = self.xyxy2xywh(bboxes[i])
                data['score'] = float(bboxes[i][4])
                data['category_id'] = 1
                json_results.append(data)
        return json_results

    def _det2json(self, results):
        """Convert detection results to COCO json style."""
        json_results = []
        for idx in range(len(self)):
            img_id = self.data_infos[idx]['id']
            result = results[idx]
            for label in range(len(result)):
                bboxes = result[label]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = self.xyxy2xywh(bboxes[i])
                    data['score'] = float(bboxes[i][4])
                    data['category_id'] = self.data_infos[idx]['obj_id']
                    json_results.append(data)
        return json_results

    def _segm2json(self, results):
        """Convert instance segmentation results to COCO json style."""
        bbox_json_results = []
        segm_json_results = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            det, seg = results[idx]
            for label in range(len(det)):
                # bbox results
                bboxes = det[label]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = self.xyxy2xywh(bboxes[i])
                    data['score'] = float(bboxes[i][4])
                    data['category_id'] = self.cat_ids[label]
                    bbox_json_results.append(data)

                # segm results
                # some detectors use different scores for bbox and mask
                if isinstance(seg, tuple):
                    segms = seg[0][label]
                    mask_score = seg[1][label]
                else:
                    segms = seg[label]
                    mask_score = [bbox[4] for bbox in bboxes]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = self.xyxy2xywh(bboxes[i])
                    data['score'] = float(mask_score[i])
                    data['category_id'] = self.cat_ids[label]
                    if isinstance(segms[i]['counts'], bytes):
                        segms[i]['counts'] = segms[i]['counts'].decode()
                    data['segmentation'] = segms[i]
                    segm_json_results.append(data)
        return bbox_json_results, segm_json_results

    def results2json(self, results, outfile_prefix):
        """Dump the detection results to a COCO style json file.

        There are 3 types of results: proposals, bbox predictions, mask
        predictions, and they have different data types. This method will
        automatically recognize the type, and dump them to json files.

        Args:
            results (list[list | tuple | ndarray]): Testing results of the
                dataset.
            outfile_prefix (str): The filename prefix of the json files. If the
                prefix is "somepath/xxx", the json files will be named
                "somepath/xxx.bbox.json", "somepath/xxx.segm.json",
                "somepath/xxx.proposal.json".

        Returns:
            dict[str: str]: Possible keys are "bbox", "segm", "proposal", and \
                values are corresponding filenames.
        """
        result_files = dict()
        if isinstance(results[0], list):
            json_results = self._det2json(results)
            # result_files['bbox'] = f'{outfile_prefix}.bbox.json'
            # result_files['proposal'] = f'{outfile_prefix}.bbox.json'
            # mmcv.dump(json_results, result_files['bbox'])
        elif isinstance(results[0], tuple):
            json_results = self._segm2json(results)
            # result_files['bbox'] = f'{outfile_prefix}.bbox.json'
            # result_files['proposal'] = f'{outfile_prefix}.bbox.json'
            # result_files['segm'] = f'{outfile_prefix}.segm.json'
            # mmcv.dump(json_results[0], result_files['bbox'])
            # mmcv.dump(json_results[1], result_files['segm'])
        elif isinstance(results[0], np.ndarray):
            json_results = self._proposal2json(results)
            # result_files['proposal'] = f'{outfile_prefix}.proposal.json'
            # mmcv.dump(json_results, result_files['proposal'])
        else:
            raise TypeError('invalid type of results')
        return json_results

    def fast_eval_recall(self, results, proposal_nums, iou_thrs, logger=None):
        gt_bboxes = []
        for i in range(len(self.img_ids)):
            ann_ids = self.coco.get_ann_ids(img_ids=self.img_ids[i])
            ann_info = self.coco.load_anns(ann_ids)
            if len(ann_info) == 0:
                gt_bboxes.append(np.zeros((0, 4)))
                continue
            bboxes = []
            for ann in ann_info:
                if ann.get('ignore', False) or ann['iscrowd']:
                    continue
                x1, y1, w, h = ann['bbox']
                bboxes.append([x1, y1, x1 + w, y1 + h])
            bboxes = np.array(bboxes, dtype=np.float32)
            if bboxes.shape[0] == 0:
                bboxes = np.zeros((0, 4))
            gt_bboxes.append(bboxes)

        recalls = eval_recalls(
            gt_bboxes, results, proposal_nums, iou_thrs, logger=logger)
        ar = recalls.mean(axis=1)
        return ar

    def format_results(self, results, jsonfile_prefix=None, **kwargs):
        """Format the results to json (standard format for COCO evaluation).

        Args:
            results (list[tuple | numpy.ndarray]): Testing results of the
                dataset.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            tuple: (result_files, tmp_dir), result_files is a dict containing \
                the json filepaths, tmp_dir is the temporal directory created \
                for saving json files when jsonfile_prefix is not specified.
        """
        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: {} != {}'.
            format(len(results), len(self)))

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None
        result_files = self.results2json(results, jsonfile_prefix)
        return result_files, tmp_dir

    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 classwise=False,
                 proposal_nums=(1, 10, 100),
                 iou_thrs=None,
                 metric_items=None):
        """Evaluation in COCO protocol.

        Args:
            results (list[list | tuple]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. Options are
                'bbox', 'segm', 'proposal', 'proposal_fast'.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            classwise (bool): Whether to evaluating the AP for each class.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thrs (Sequence[float], optional): IoU threshold used for
                evaluating recalls/mAPs. If set to a list, the average of all
                IoUs will also be computed. If not specified, [0.50, 0.55,
                0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95] will be used.
                Default: None.
            metric_items (list[str] | str, optional): Metric items that will
                be returned. If not specified, ``['AR@100', 'AR@300',
                'AR@1000', 'AR_s@1000', 'AR_m@1000', 'AR_l@1000' ]`` will be
                used when ``metric=='proposal'``, ``['mAP', 'mAP_50', 'mAP_75',
                'mAP_s', 'mAP_m', 'mAP_l']`` will be used when
                ``metric=='bbox' or metric=='segm'``.

        Returns:
            dict[str, float]: COCO style evaluation metric.
        """

        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['bbox', 'segm', 'proposal', 'proposal_fast']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')
        if iou_thrs is None:
            iou_thrs = np.linspace(
                .5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        if metric_items is not None:
            if not isinstance(metric_items, list):
                metric_items = [metric_items]

        json_results, _ = self.format_results(results, jsonfile_prefix)

        eval_results = OrderedDict()
        cocoGt = self.coco
        for metric in metrics:
            msg = f'Evaluating {metric}...'
            if logger is None:
                msg = '\n' + msg
            print_log(msg, logger=logger)
            cocoDt = cocoGt.loadRes(json_results)

            iou_type = 'bbox' if metric == 'proposal' else metric
            cocoEval = COCOeval(cocoGt, cocoDt, iou_type)
            cocoEval.params.catIds = self.cat_ids
            cocoEval.params.imgIds = self.img_ids
            cocoEval.params.maxDets = [1]
            cocoEval.params.iouThrs = np.array([0.25, 0.5 , 0.55, 0.6 , 0.65, 0.7 , 0.75, 0.8 , 0.85, 0.9 , 0.95])

            cocoEval.evaluate()
            cocoEval.accumulate()
            stats = self.summarize(cocoEval.eval['precision'], cocoEval.eval['recall'], cocoEval.params)

        if metric_items is None:
            metric_items = [
                    'mAR', 'AR_25', 'AR_50', 'AR_75'
                ]

        for i, metric_item in enumerate(metric_items):
            key = f'{metric}_{metric_item}'
            val = float(
                f'{stats[i]:.3f}'
            )
            eval_results[key] = val
        ar = stats[:4]
        eval_results[f'{metric}_mAP_copypaste'] = (
            f'{ar[0]:.3f} {ar[1]:.3f} {ar[2]:.3f} {ar[3]:.3f} ')

        return eval_results

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

    def cam2pose_rel_q(self, cam, query_rotation):
        # relative to the first frame
        rela_r = torch.eye(3).unsqueeze(0).repeat(1, 1, 1)
        rela_t = torch.zeros((1, 3, 1))
        pre_r = torch.Tensor(cam[str(0)][0]["cam_R_m2c"]).reshape(3, 3)
        curr_r = torch.Tensor(query_rotation[0])
        rela_r[0] = curr_r @ pre_r.T
        rela_pose = torch.cat([rela_r, rela_t], dim=-1)
        return rela_pose

    def summarize(self, precision, recall, params):
        '''
        Compute and display summary metrics for evaluation results.
        Note this functin can *only* be applied on the default parameter
        setting
        '''
        def _summarize(precision, recall, params, ap=1, iouThr=None, areaRng='all', maxDets=100):
            p = params
            iStr = '{:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'  # noqa: E501
            titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
            typeStr = '(AP)' if ap == 1 else '(AR)'
            iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[1], p.iouThrs[-1]) \
                if iouThr is None else '{:0.2f}'.format(iouThr)

            aind = [
                i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng
            ]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
            if ap == 1:
                # dimension of precision: [TxRxKxAxM]
                s = precision
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                else:
                    s = s[1:, :, :, aind, mind]
            else:
                # dimension of recall: [TxKxAxM]
                s = recall
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                else:
                    s = s[1:, :, aind, mind]
            if len(s[s > -1]) == 0:
                mean_s = -1
            else:
                mean_s = np.mean(s[s > -1])
            print(
                iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets,
                            mean_s))
            return mean_s

        stats = np.zeros((10, ))
        stats[0] = _summarize(precision, recall, params, 0, maxDets=1)
        stats[1] = _summarize(precision, recall, params, 0, iouThr=.25, maxDets=1)
        stats[2] = _summarize(precision, recall, params, 0, iouThr=.5, maxDets=1)
        stats[3] = _summarize(precision, recall, params, 0, iouThr=.75, maxDets=1)
        stats[4] = _summarize(precision, recall, params, 0,
                                areaRng='small',
                                maxDets=1)
        stats[5] = _summarize(precision, recall, params, 0,
                                areaRng='medium',
                                maxDets=1)
        stats[6] = _summarize(precision, recall, params, 0,
                                areaRng='large',
                                maxDets=1)
        return stats
