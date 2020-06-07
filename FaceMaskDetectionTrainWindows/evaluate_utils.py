# -*- encoding:utf-8 -*-
import numpy as np
from tqdm import tqdm

'''
思路:
1. 标签按照以下格式存放为一个文件，读取为列表，每个列表为一行ground truth
<img_name> <class_name> <xmin> <ymin> <xmax> <ymax> <difficult_flag>

2. 预测结果按照以下格式存放，读取为列表，每个列表为一行ground truth
<img_name> <class_name> <confidence> <xmin> <ymin> <xmax> <ymax> 

整体合并为一个numpy的array，注意，如果img_name来自不同文件夹，有重名的可能，可以使用全部路径，这样也可以唯一定位图片。
这样的array，类型是'|s7',使用其他位置信息或者flag信息的时候，注意在评估函数里面转换格式；

3. 评估的时候，对于每个类别，分别统计每个类的Average Precision，然后再统计mAP
'''

def voc_ap(rec, prec, use_07_metric=False):
    """Compute VOC AP given precision and recall. If use_07_metric is true, uses
    the VOC 07 11-point method, otherwise use 2011 metric (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def calculate_ap(cate_gt,
                 cate_predicted,
                 omit_difficult,
                 iou_thresh=0.5,
                 use_07_metric=True):
    '''
    1. 标签按照以下格式存放为一个文件，读取为列表，每个列表为一行ground truth
    <img_name> <class_name> <xmin> <ymin> <xmax> <ymax> <difficult_flag>

    2. 预测结果按照以下格式存放，读取为列表，每个列表为一行ground truth
    <img_name> <class_name> <confidence> <xmin> <ymin> <xmax> <ymax>
    :param cate_gt:
    :param cate_predicted:
    :param omit_difficult:
    :param iou_thresh:
    :param use_07_metric:
    :return:
    '''
    predict_conf = cate_predicted[:, 2].astype(np.float32)
    predict_conf_args = np.argsort(-predict_conf)

    # num_positive = 0
    gt_flags = {} # 使用一个字典类型作为标志量，其中key为图片名称，value为一个列表，长度与该图片的ground truth数目一致
    difficult_flags = {}

    for image_name in list(set(cate_gt[:, 0])):
        gt_flags[image_name] = [0] * (cate_gt[:, 0] == image_name).sum()
        if cate_gt.shape[1] == 6:
            difficult_flags[image_name] = [0] * (cate_gt[:, 0] == image_name).sum()
        else:
            difficult_flags[image_name] = ((cate_gt[cate_gt[:, 0] == image_name])[:, 6]).astype(np.int8)

    if omit_difficult:
        if cate_gt.shape[1] != 7:
            raise ValueError("You didn't supply difficult flag.")
        num_positive = len(cate_gt) - cate_gt[:, 6].astype(np.float32).sum()
    else:
        num_positive = len(cate_gt)

    tp = np.zeros(len(cate_predicted))
    fp = np.zeros(len(cate_predicted))

    for idx, arg_idx in tqdm(enumerate(predict_conf_args)):
        image_name = cate_predicted[arg_idx, 0]
        bbox = (cate_predicted[arg_idx, 3:]).astype(np.float32)
        gt_bboxes = (cate_gt[cate_gt[:, 0] == image_name][:, 2:6]).astype(np.float32)

        if len(gt_bboxes) == 0:
            fp[idx] = 1
            continue
        ixmin = np.maximum(gt_bboxes[:, 0], bbox[0])
        iymin = np.maximum(gt_bboxes[:, 1], bbox[1])
        ixmax = np.minimum(gt_bboxes[:, 2], bbox[2])
        iymax = np.minimum(gt_bboxes[:, 3], bbox[3])
        iw = np.maximum(ixmax - ixmin + 1., 0.)
        ih = np.maximum(iymax - iymin + 1., 0.)
        inters = iw * ih
        # union
        uni = (bbox[2] - bbox[0] + 1.) * (bbox[3] - bbox[1] + 1.) +\
               (gt_bboxes[:, 2] - gt_bboxes[:, 0] + 1.) *(gt_bboxes[:, 3] - gt_bboxes[:, 1] + 1.) - inters

        overlaps = inters / uni
        ovmax = np.max(overlaps)
        jmax = np.argmax(overlaps)

        if ovmax > iou_thresh:
            if omit_difficult: # 忽略困难样本模式
                if cate_gt.shape[1] != 7:
                    raise ValueError("You didn't supply difficult flag.")

                if difficult_flags[image_name][jmax] == 0:  # 只有非困难的样本才会参与计算， 如果是困难的bbox，则忽略掉这个
                    # 非困难样本
                    if not gt_flags[image_name][jmax]:
                        tp[idx] = 1
                        gt_flags[image_name][jmax] = 1
                    else:
                        fp[idx] = 1

            else:  # 非忽略困难样本模式
                if not gt_flags[image_name][jmax]:
                    tp[idx] = 1
                    gt_flags[image_name][jmax] = 1
                else:
                    fp[idx] = 1
        else:
            fp[idx] = 1

    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    recall = tp / float(num_positive)
    precsion = tp / np.maximum(tp + fp, 1)
    ap = voc_ap(recall, precsion, use_07_metric=use_07_metric)
    return {'precision': precsion, 'recall':recall, 'ap': ap}


def evaluate(gt_info, predicted_info, omit_difficult=True, iou_thresh=0.5, use_07_metric=False):
    '''
    1. 标签按照以下格式存放为一个文件，读取为列表，每个列表为一行ground truth
    <img_name> <class_name> <xmin> <ymin> <xmax> <ymax> <difficult_flag>

    2. 预测结果按照以下格式存放，读取为列表，每个列表为一行ground truth
    <img_name> <class_name> <confidence> <xmin> <ymin> <xmax> <ymax>

    :param gt_info:
    :param predicted_info:
    :return:
    '''
    gt_info = np.array(gt_info)
    predicted_info = np.array(predicted_info)

    categories = list(set(gt_info[:, 1]))
    result = {}
    for cate in categories:
        cate_gt = gt_info[gt_info[:, 1] == cate]
        cate_predicted = predicted_info[predicted_info[:, 1] == cate]
        # print(cate_gt, cate_predicted)
        cate_result = calculate_ap(cate_gt,
                                   cate_predicted,
                                   omit_difficult=omit_difficult,
                                   iou_thresh=iou_thresh,
                                   use_07_metric=use_07_metric
                                   )
        result[cate] = cate_result
    return result




