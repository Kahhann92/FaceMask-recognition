# -*- encoding=utf-8 -*-
import numpy as np
from tqdm import tqdm

def voc_ap(rec, prec, use_07_metric=False):
    """Compute VOC AP given precision and recall. If use_07_metric is true, uses
    the VOC 07 11-point method (default:False).
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


def pascal_voc_eval(predict_info,
                    gt_info,
                    difficult_flags,
                    omit_difficult=True,
                    iou_thresh=0.5,
                    use_07_metric=False):
    '''
    :param predict_info: 2D array of the predict results, [number_predictions, 6], each line represents
           [ image_id, conf, xmin, ymin, xmax, ymax]
    :param gt_info: a dict, the key is the image id, while the value is 2D numpy array--ground true bbox,
           {image_id:nd.array}
    :param difficult_flags:  a dict, the key is the image id, while the value is a list,
            indicate the difficult flag corresponding to bbox, '0' is normal, '1' is difficult.
    :param omit_difficult:
    :param iou_thresh:
    :param use_07_metric:
    :return:
        precision
        recall
        ap
    '''
    predict_conf = predict_info[:, 1]
    predict_belonging_image_id = predict_info[:, 0]
    predict_bboxes = predict_info[:, 2:]
    predict_conf_args = np.argsort(-predict_conf)
    # 标记每个ground truth是否已经被正确预测了，被正确预测过就设置为1， 否则为0
    # mark each ground truth bbox whether has correct predict bbox or not.
    gt_flags = {}
    npos = 0

    # 初始化为0， 也就是都没正确预测过
    for k in difficult_flags.keys():
        gt_flags[k] = [0] * len(difficult_flags[k])
        if omit_difficult:
            # 困难样本标记为1， 正常样本标记为0， 所以用1 - int(x) 来统计正常样本的个数
            npos += sum([(1 - int(x)) for x in difficult_flags[k]])
        else:
            npos += len(difficult_flags[k])

    print("Total number of positive %d" % npos)
    tp = np.zeros(len(predict_info))
    fp = np.zeros(len(predict_info))

    for i, arg_idx in tqdm(enumerate(predict_conf_args)):
        ovmax = -np.inf
        image_id = predict_belonging_image_id[arg_idx]
        bbox = predict_bboxes[arg_idx, :]
        try:
            gt_bboxes = gt_info[int(image_id)]
        except KeyError:
            fp[i] = 1
            continue

        if len(gt_bboxes) > 0:
            # compute overlaps
            ixmin = np.maximum(gt_bboxes[:, 0], bbox[0])
            iymin = np.maximum(gt_bboxes[:, 1], bbox[1])
            ixmax = np.minimum(gt_bboxes[:, 2], bbox[2])
            iymax = np.minimum(gt_bboxes[:, 3], bbox[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih
            # union
            uni = ((bbox[2] - bbox[0] + 1.) * (bbox[3] - bbox[1] + 1.) +
                   (gt_bboxes[:, 2] - gt_bboxes[:, 0] + 1.) *
                   (gt_bboxes[:, 3] - gt_bboxes[:, 1] + 1.) - inters)

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > iou_thresh:
            if omit_difficult: # 忽略困难样本模式
                if difficult_flags[image_id][jmax] == '0':  # 只有非困难的样本才会参与计算， 如果是困难的bbox，则忽略掉这个
                    if not gt_flags[image_id][jmax]:
                        tp[i] = 1
                        gt_flags[image_id][jmax] = 1
                    else:
                        fp[i] = 1

            else:  # 非忽略困难样本模式
                if not gt_flags[image_id][jmax]:
                    tp[i] = 1
                    gt_flags[image_id][jmax] = 1
                else:
                    fp[i] = 1
        else:
            fp[i] = 1

    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    recall = tp / float(npos)
    precsion = tp / np.maximum(tp + fp, 1)
    ap = voc_ap(recall, precsion, use_07_metric=use_07_metric)
    return precsion, recall, ap