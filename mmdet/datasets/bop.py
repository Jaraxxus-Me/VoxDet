import itertools
import logging
import os.path as osp
import tempfile
from collections import OrderedDict
import os
import torch
import copy
import json
from glob import glob

import mmcv
import numpy as np
from mmcv.utils import print_log
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from bop_toolkit_lib import pycoco_utils
import argparse

from bop_toolkit_lib import config
from bop_toolkit_lib import dataset_params
from bop_toolkit_lib import inout
from bop_toolkit_lib import misc

from terminaltables import AsciiTable

from mmdet.core import eval_recalls
from .builder import DATASETS
from .coco import CocoDataset
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

LMO_OBJECT = ('1', '5', '6', '8', '9',
            '10', '11', '12')
YCB_CLASSES = ('1', '2', '3', '4', '5',
            '6', '7', '8', '9', '10', '11', '12',
            '13', '14', '15', '16', '17', '18', '19',
            '20', '21')
ROBO_CLASSES = ('1', '2', '3', '4', '5',
            '6', '7', '8', '9', '10', '11', '12',
            '13', '14', '15', '16', '17', '18', '19',
            '20')
MAX_DET = 1

@DATASETS.register_module()
class BopDataset(CustomDataset):

    def __init__(self,
                 p1_path=None,
                 **kwargs):
        # We convert all category IDs into 1 for the class-agnostic training and
        # evaluation. We train on train_class and evaluate on eval_class split.
        self.support = True if p1_path != None else False
        self.p1_path = p1_path
        object_names = {
            'lmo': LMO_OBJECT,
            'ycbv': YCB_CLASSES,
            'RoboTools': ROBO_CLASSES,
            'RoboTools_special': ROBO_CLASSES
        }
        # camera pose info
        all_cam_path = os.path.join(self.p1_path, 'scene_gt_all.json')
        if os.path.isfile(all_cam_path):
            with open (all_cam_path, 'r') as f:
                self.cam_pos = json.load(f)
        else:
            print('Start constructing object camera and size')
            self.cam_pos = {}
            video_paths = sorted(glob(self.p1_path+'/*/'))
            for v in video_paths:
                obj_id = int(v.split('/')[-2][-2:])
                cam_path = os.path.join(v, 'scene_gt.json')
                with open (cam_path, 'r') as f:
                    self.cam_pos[obj_id] = json.load(f)
            with open (all_cam_path, 'w') as f:
                json.dump(self.cam_pos, f)
        dataset_name = kwargs['ann_file'].split('/')[2]
        self.CLASSES = object_names[dataset_name]
        super(BopDataset, self).__init__(classes = self.CLASSES, **kwargs)


    def load_annotations(self, ann_file):
        """Load annotation from COCO style annotation file.

        Args:
            ann_file (str): Path of annotation file.

        Returns:
            list[dict]: Annotation info from COCO api.
        """

        self.coco = COCO(ann_file)
        self.cat_ids = self.coco.get_cat_ids(cat_names=self.CLASSES)
        self.cat2label = {
            str(cat_id): cat_id for i, cat_id in enumerate(self.cat_ids)}

        self.img_ids = self.coco.get_img_ids()
        data_infos = []
        if self.support:
            # if there is support, dataset is one instance per img
            for i in self.img_ids:
                id_list = []
                info = self.coco.load_imgs([i])[0]
                anns = self.coco.getAnnIds(info['id'])
                for i, ann in enumerate(self.coco.load_anns(anns)):
                    ann['filename'] = info['file_name']
                    data_infos.append(ann)
        else:
            # if no support, dataset is split by img
            for i in self.img_ids:
                id_list = []
                info = self.coco.load_imgs([i])[0]
                info['filename'] = info['file_name']
                for key in self.coco.cat_img_map.keys():
                    if info['id'] in self.coco.cat_img_map[key]:
                        id_list.append(key)
                info['id_list'] = id_list
                data_infos.append(info)
        return data_infos

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
        p1_data = np.load(os.path.join(self.p1_path, 'obj_{:06d}'.format(img_info['category_id']), 'info.npz'))
        data['rgb'] = torch.from_numpy(p1_data['rgb'].astype(np.float32))
        data['mask'] = torch.from_numpy(p1_data['mask'].astype(np.float32))
        # data['point'] = torch.from_numpy(p1_data['point'].astype(np.float32))
        imgids = list(range(160))
        if 'Robo' in self.p1_path:
            imgids = list(range(100))
        cam_traj = self.cam2pose_rel(self.cam_pos["obj_{:06d}".format(img_info['category_id'])], imgids)
        data['traj'] = cam_traj
        data['id'] = img_info['category_id']
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
        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.train_cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)                
                gt_labels.append(self.cat2label[ann['category_id']])
                gt_masks_ann.append(ann.get('segmentation', None))

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
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            seg_map=seg_map)

        return ann

    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 proposal_nums=(10, 20, 30, 50, 100, 300, 500, 1000, 1500),
                 iou_thrs=None,
                 metric_items=None):
        """Evaluation in COCO-Split protocol.

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

        result_files, tmp_dir = self.format_results(results, jsonfile_prefix)

        eval_results = OrderedDict()
        cocoGt = self.coco
        for metric in metrics:
            msg = f'Evaluating {metric}...'
            if logger is None:
                msg = '\n' + msg
            print_log(msg, logger=logger)

            if metric == 'proposal_fast':
                ar = self.fast_eval_recall(
                    results, proposal_nums, iou_thrs, logger='silent')
                log_msg = []
                for i, num in enumerate(proposal_nums):
                    eval_results[f'AR@{num}'] = ar[i]
                    log_msg.append(f'\nAR@{num}\t{ar[i]:.4f}')
                log_msg = ''.join(log_msg)
                print_log(log_msg, logger=logger)
                continue

            if metric not in result_files:
                raise KeyError(f'{metric} is not in results')
            try:
                cocoDt = cocoGt.loadRes(result_files[metric])
            except IndexError:
                print_log(
                    'The testing results of the whole dataset is empty.',
                    logger=logger,
                    level=logging.ERROR)
                break

            iou_type = 'bbox' if metric == 'proposal' else metric


            # Cross-category evaluation wrapper.
            # running evaluation
            cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
            cocoEval.params.imgIds = sorted(cocoGt.getImgIds())
            cocoEval.params.maxDets = [MAX_DET]
            cocoEval.params.iouThrs = np.array([0.25, 0.5 , 0.55, 0.6 , 0.65, 0.7 , 0.75, 0.8 , 0.85, 0.9 , 0.95])
            cocoEval.evaluate()
            cocoEval.accumulate()
            stats = self.summarize(cocoEval.eval['precision'], cocoEval.eval['recall'], cocoEval.params)
            # Compute per-category AP
            # from https://github.com/facebookresearch/detectron2/
            precisions = cocoEval.eval['precision']
            recalls = cocoEval.eval['recall']
            # precision: (iou, recall, cls, area range, max dets)
            assert len(self.cat_ids) == precisions.shape[2]
            assert len(self.cat_ids) == recalls.shape[1]

            results_per_category = []
            for idx, catId in enumerate(self.cat_ids):
                # area range index 0: all area ranges
                # max dets index 0: typically 1 for every instance per image 
                nm = self.coco.loadCats(catId)[0]
                recall = recalls[:, idx, 0, 0]
                if recall.size:
                    ar = np.mean(recall)
                else:
                    ar = float('nan')
                results_per_category.append(
                    (f'{nm["name"]}', f'{float(ar):0.3f}'))

            num_columns = min(8, len(results_per_category) * 2)
            results_flatten = list(
                itertools.chain(*results_per_category))
            headers = ['category', 'mAR@1'] * (num_columns // 2)
            results_2d = itertools.zip_longest(*[
                results_flatten[i::num_columns]
                for i in range(num_columns)
            ])
            table_data = [headers]
            table_data += [result for result in results_2d]
            table = AsciiTable(table_data)
            print_log('\n' + table.table, logger=logger)
            # mapping of cocoEval.stats
        if tmp_dir is not None:
            tmp_dir.cleanup()
        return eval_results

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
        obj_ids = None
        if isinstance(results, list): # 'results from OLN_RPN is a list: [[ndarray for img1], [ndarray for img2], ...]'
            assert len(results) == len(self), (
                'The length of results is not equal to the dataset len: {} != {}'.
                format(len(results), len(self)))
            result_files={}
            if jsonfile_prefix is None:
                tmp_dir = tempfile.TemporaryDirectory()
                jsonfile_prefix = osp.join(tmp_dir.name, 'results')
                result_files = self.results2json(results, jsonfile_prefix)
            else:
                tmp_dir = None
                if os.path.isfile(f'{jsonfile_prefix}.bbox.json'):
                    result_files['bbox'] = f'{jsonfile_prefix}.bbox.json'
                    result_files['proposal'] = f'{jsonfile_prefix}.bbox.json'
                    return result_files, tmp_dir
            result_files = self.results2json(results, jsonfile_prefix)
        if isinstance(results, dict): # 'results from other methods are a dict: {'img1_id': [[list(ndarray) for img1]}'
            result_files={}
            if jsonfile_prefix is None:
                tmp_dir = tempfile.TemporaryDirectory()
                jsonfile_prefix = osp.join(tmp_dir.name, 'results')
            else:
                tmp_dir = None
                if os.path.isfile(f'{jsonfile_prefix}.bbox.json'):
                    result_files['bbox'] = f'{jsonfile_prefix}.bbox.json'
                    result_files['proposal'] = f'{jsonfile_prefix}.bbox.json'
                    return result_files, tmp_dir
            result_files = self.results2json(results, jsonfile_prefix, obj_ids)
        return result_files, tmp_dir

    def results2json(self, results, outfile_prefix, obj_ids=None):
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
        json_results = []
        for idx in range(len(self)):
            if isinstance(results, list):
                result = results[idx]
            else:
                result = results[idx+1]
            # one image should have [num_of_target_instances] output boxes, corresponding to the instance ids in the image
            if isinstance(results, list):
                bboxes = result[0]
            else:
                bboxes = np.array(result)
            img_id = self.data_infos[idx]['image_id']
            assert img_id==(idx+1)
            id = self.data_infos[idx]['category_id']
            for i in range(bboxes.shape[0]):
                data = dict()
                data['image_id'] = img_id
                data['bbox'] = self.xyxy2xywh(bboxes[i])
                data['score'] = float(bboxes[i][4])
                data['category_id'] = id
                json_results.append(data)
        result_files['bbox'] = f'{outfile_prefix}.bbox.json'
        result_files['proposal'] = f'{outfile_prefix}.bbox.json'
        mmcv.dump(json_results, result_files['bbox'])
        return result_files

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
        stats[0] = _summarize(precision, recall, params, 0, maxDets=MAX_DET)
        stats[1] = _summarize(precision, recall, params, 0, iouThr=.25, maxDets=MAX_DET)
        stats[2] = _summarize(precision, recall, params, 0, iouThr=.5, maxDets=MAX_DET)
        stats[3] = _summarize(precision, recall, params, 0, iouThr=.75, maxDets=MAX_DET)
        stats[4] = _summarize(precision, recall, params, 0,
                                areaRng='small',
                                maxDets=MAX_DET)
        stats[5] = _summarize(precision, recall, params, 0,
                                areaRng='medium',
                                maxDets=MAX_DET)
        stats[6] = _summarize(precision, recall, params, 0,
                                areaRng='large',
                                maxDets=MAX_DET)
        return stats
    
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
