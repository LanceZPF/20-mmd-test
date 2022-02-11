import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import numpy as np
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model
from numpy.core.defchararray import count
from pycocotools.cocoeval import COCOeval

from mmdet.apis import single_gpu_test
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.models import build_detector

# Basic settings. Later these would be moved to argparse.
import argparse
parser = argparse.ArgumentParser(description="Input the path of model configuration and model weights to evaluate them with new indicator.")
parser.add_argument("--config_path",type=str,default = './logs_dy/dynamic_rcnn_r50_fpn_1x.py',help="path of the model configuration.")
parser.add_argument("--weight_path",type=str,default = './logs_dy/epoch_42.pth',help="path of the model weights.")

args=parser.parse_args()
cfg_path=args.config_path
ckpt_path=args.weight_path

#cfg_path="fcos_dxcn.py"
#ckpt_path="./work_dirs/fcos_dxcn/epoch_180.pth"
# should set 0 if consider classification only?
pos_score_threshold = 0.5
min_iou_threshold = 0.5

# Load configuration file with Config from mmcv
cfg = Config.fromfile(cfg_path)

######## The following code is copied from tools/test.py ########
# But no distribution
# Extra steps for rfp_backbone
if cfg.model.get('neck'):
    if isinstance(cfg.model.neck, list):
        for neck_cfg in cfg.model.neck:
            if neck_cfg.get('rfp_backbone'):
                if neck_cfg.rfp_backbone.get('pretrained'):
                    neck_cfg.rfp_backbone.pretrained = None
    elif cfg.model.neck.get('rfp_backbone'):
        if cfg.model.neck.rfp_backbone.get('pretrained'):
            cfg.model.neck.rfp_backbone.pretrained = None

# in case the test dataset is concatenated
samples_per_gpu = 1
if isinstance(cfg.data.test, dict):
    cfg.data.test.test_mode = True
    samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
    if samples_per_gpu > 1:
        # Replace 'ImageToTensor' to 'DefaultFormatBundle'
        cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
elif isinstance(cfg.data.test, list):
    for ds_cfg in cfg.data.test:
        ds_cfg.test_mode = True
    samples_per_gpu = max(
        [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
    if samples_per_gpu > 1:
        for ds_cfg in cfg.data.test:
            ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

# This do not support dist_test now.
distributed = False

# build the dataloader
dataset = build_dataset(cfg.data.test)
data_loader = build_dataloader(
    dataset,
    samples_per_gpu=samples_per_gpu,
    workers_per_gpu=cfg.data.workers_per_gpu,
    dist=distributed,
    shuffle=False)

# build the model and load checkpoint
cfg.model.train_cfg = None
model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
fp16_cfg = cfg.get('fp16', None)
if fp16_cfg is not None:
    wrap_fp16_model(model)
checkpoint = load_checkpoint(model, ckpt_path, map_location='cpu')
# old versions did not save class info in checkpoints, this walkaround is
# for backward compatibility
if 'CLASSES' in checkpoint.get('meta', {}):
    model.CLASSES = checkpoint['meta']['CLASSES']
else:
    model.CLASSES = dataset.CLASSES

model = MMDataParallel(model, device_ids=[0])
outputs = single_gpu_test(model, data_loader)

##################### test.py Copy End ######################

####################FOR TEST ONLY NOW DEPRECATED####################
if False:
    import pickle
    with open("test_output.pkl", mode='wb') as f:
        pickle.dump(outputs, f)
    exit()

    import pickle
    with open("test_output.pkl", mode='rb') as f:
        outputs = pickle.load(f)
#################FOR TEST ONLY NOW DEPRECATED END####################

########### The Following code is modified from CocoDataset.evaluate

result_files, tmp_dir = dataset.format_results(outputs)

# fAcc for full accuracy.
# AER for Average Error Rate.
eval_results = {}
metric = "bbox"
cocoGt = dataset.coco
msg = f'Evaluating {metric}...'
try:
    cocoDt = cocoGt.loadRes(result_files[metric])
except IndexError:
    print('The testing results of the whole dataset is empty.')

cocoEval = COCOeval(cocoGt, cocoDt, "bbox")
cocoEval.params.catIds = dataset.cat_ids
cocoEval.params.imgIds = dataset.img_ids

# COCO AP EVALUATION
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()

p = cocoEval.params
p.imgIds = list(np.unique(p.imgIds))
p.catIds = list(np.unique(p.catIds))
p.maxDets = sorted(p.maxDets)
p.useCats = False
cocoEval.params = p
cocoEval._prepare()
catIds = p.catIds
ious = {
    imgId: cocoEval.computeIoU(imgId, catId)
    for catId in catIds for imgId in p.imgIds
}
maxDet = p.maxDets[-1]
fAcc = 0
AER = 0
num_instances = 0

sortxyxy = lambda x: [
    min(x['bbox'][0], x['bbox'][2]),
    min(x['bbox'][1], x['bbox'][3]),
    max(x['bbox'][0], x['bbox'][2]),
    max(x['bbox'][1], x['bbox'][3])
]


def get_ious(lbox, rbox):
    ious_mat = np.zeros((len(lbox), len(rbox)))
    if len(lbox) * len(rbox) == 0:
        return ious_mat
    lbox = list(map(sortxyxy, lbox))
    rbox = list(map(sortxyxy, rbox))
    lx1s = [x[0] for x in lbox]
    lx2s = [x[2] for x in lbox]
    rx1s = [x[0] for x in rbox]
    rx2s = [x[2] for x in rbox]
    ly1s = [x[1] for x in lbox]
    ly2s = [x[3] for x in lbox]
    ry1s = [x[1] for x in rbox]
    ry2s = [x[3] for x in rbox]
    sx1s = [lx2s[l] - lx1s[l] for l in range(len(lbox))]
    sy1s = [ly2s[l] - ly1s[l] for l in range(len(lbox))]
    sx2s = [rx2s[r] - rx1s[r] for r in range(len(rbox))]
    sy2s = [ry2s[r] - ry1s[r] for r in range(len(rbox))]
    #print("params:")
    #print(lbox, rbox)
    for l in range(len(lbox)):
        for r in range(len(rbox)):
            iou_x = min(lx1s[l], lx2s[l], rx1s[r], rx2s[r]) - max(
                lx1s[l], lx2s[l], rx1s[r], rx2s[r]) + sx1s[l] + sx2s[r]
            iou_y = min(ly1s[l], ly2s[l], ry1s[r], ry2s[r]) - max(
                ly1s[l], ly2s[l], ry1s[r], ry2s[r]) + sy1s[l] + sy2s[r]
            iou_x, iou_y = max(0, iou_x), max(0, iou_y)
            ious_mat[l, r] = (iou_y * iou_x) / (
                sx1s[l] * sy1s[l] + sx2s[r] * sy2s[r] - iou_x * iou_y)
    return ious_mat

def match_ious(matrix: list, iou_thre: float) -> list:
    """Match rows with cols.

    Args:
        matrix (list): iou matrix in shape (dt,gt)
        iou (float): iou threshold
    Returns:
        list: match results.
    """
    # Empty
    if len(matrix) == 0 or len(matrix[0]) == 0: return [], []
    dt = [-1 for _ in matrix]
    gt = [-1 for _ in matrix[0]]
    # 依次阈值计算？
    for dind, dline in enumerate(matrix):
        # 依次枚举待预测d.m代表可能最适被匹配的g下标。
        # information about best match so far (m=-1 -> unmatched)
        m, iou = -1, iou_thre
        # 依次匹配gt
        for gind, dg_iou in enumerate(dline):
            # 如果gt已匹配，跳过
            if gt[gind] > -1:
                continue
            # 找到最大的未已经匹配的匹配
            if dg_iou > iou:
                # if match successful and best so far, store appropriately
                iou = dg_iou
                m = gind
        # if match made store id of match for both dt and gt
        if m != -1:
            dt[dind] = m
            gt[m] = dind
    return dt, gt


def stat_single_img(p, ious, seq_num, score_thre, iou_thre):
    gt = [_ for cId in p.catIds for _ in cocoEval._gts[p.imgIds[seq_num], cId]]
    dt = [_ for cId in p.catIds for _ in cocoEval._dts[p.imgIds[seq_num], cId]]
    dt_ind = np.argsort([-d['score'] for d in dt], kind='mergesort')
    dnum = min(len([d for d in dt if d['score'] > score_thre]), maxDet)

    # use results from pycocotools. but seems to be ... weird.
    pos_ious = [ious[p.imgIds[seq_num]][i, :] for i in dt_ind[:dnum]]
    #Mine. Not fixed Yet.
    dt_d = [dt[i] for i in dt_ind[:dnum]]
    pos_ious = get_ious(dt_d, gt)

    #print("----------------------")
    #print(dt_ind[:dnum])
    #print(np.array(pos_ious))
    #print("dt_boxes:",
    #      [[int(x) for x in dt[i]["bbox"]] for i in dt_ind[:dnum]])
    #print("gt_boxes:", [[int(x) for x in g["bbox"]] for g in gt])
    #print([dt[i]['category_id'] for i in dt_ind[:dnum]])
    #print([g['category_id'] for g in gt])
    dm_ind, gm_ind = match_ious(pos_ious, iou_thre)
    # 对空预测做特殊处理
    if len(gm_ind) == 0 and len(gt) != 0:
        return len(gt)
    # 统计空匹配数目
    #print(dm_ind, gm_ind)
    empty = dm_ind.count(-1) + gm_ind.count(-1)
    # from pycocotools.
    #mispred = [
    #    dt[dt_ind[i]]['category_id'] == gt[gt_id]['category_id']
    #    for (i, gt_id) in enumerate(dm_ind) if gt_id >= 0
    #].count(False)

    # mine
    mispred = [
        dt_d[i]['category_id'] == gt[gt_id]['category_id']
        for (i, gt_id) in enumerate(dm_ind) if gt_id >= 0
    ].count(False)
    #print("empty=%d;mispred=%d" % (empty, mispred))
    return empty + mispred


print("Save outputs...")
import pickle
with open("coco_eval.pickle", mode='wb') as f:
    pickle.dump(cocoEval, f)

print("Evaluating FAcc and AER%d..." % (int(min_iou_threshold * 100)))
for i in range(len(p.imgIds)):
    single_AER = stat_single_img(p, ious, i, pos_score_threshold,
                                 min_iou_threshold)
    if single_AER == 0:
        fAcc += 1
    AER += single_AER
    # if it's needed to  stat average instances num.
    #num_instances += len(
    #    [_ for cId in p.catIds for _ in cocoEval._gts[p.imgIds[i], cId]])

fAcc /= len(p.imgIds)
#AER over images
AER /= len(p.imgIds)

#AER over instances
#AER /=num_instances

print("fAcc(IoUThreshold@%.2f) = %.4f" % (min_iou_threshold, fAcc))
print("AER(IoUThreshold@%.2f) = %.4f" % (min_iou_threshold, AER))
