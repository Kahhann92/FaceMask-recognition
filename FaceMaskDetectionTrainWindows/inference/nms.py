# -*- encoding=utf-8
import numpy as np


def single_class_non_max_suppression(bboxes,
                                     confidences,
                                     conf_thresh=0.2,
                                     iou_thresh=0.5,
                                     keep_top_k=-1,
                                     softnms="gaussian",
                                     sigma=0.1
                                     ):
    '''
    do nms on single class
    算法思路：对于给定的某一类的bbox，以及该类的bbox对应的置信度，按照置信度排序，置信度从高到低，依次选置信度最高的bbox，
    与其他剩余的bbox做iou计算，滤除iou大于阈值的bbox，把这些需要滤出的从排序列表中删除，然后依次循环，直到列表中待选的bbox数量为零。
    :param bboxes: numpy array of 2D, [num_bboxes, 4]
    :param confidences: numpy array of 1D. [num_bboxes]
    :param conf_thresh:
    :param iou_thresh:
    :param keep_top_k:
    :return:
    '''
    if len(bboxes) == 0: return []

    conf_keep_idx = np.where(confidences > conf_thresh)[0]

    bboxes = bboxes[conf_keep_idx]
    confidences = confidences[conf_keep_idx]

    pick = []
    xmin = bboxes[:, 0]
    ymin = bboxes[:, 1]
    xmax = bboxes[:, 2]
    ymax = bboxes[:, 3]

    area = (xmax - xmin + 1e-3) * (ymax - ymin + 1e-3)
    idxs = np.argsort(confidences)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        current_confidence = confidences[idxs[:last]]

        # keep top k
        if keep_top_k != -1:
            if len(pick) >= keep_top_k:
                break

        overlap_xmin = np.maximum(xmin[i], xmin[idxs[:last]])
        overlap_ymin = np.maximum(ymin[i], ymin[idxs[:last]])
        overlap_xmax = np.minimum(xmax[i], xmax[idxs[:last]])
        overlap_ymax = np.minimum(ymax[i], ymax[idxs[:last]])
        overlap_w = np.maximum(0, overlap_xmax - overlap_xmin)
        overlap_h = np.maximum(0, overlap_ymax - overlap_ymin)
        overlap_area = overlap_w * overlap_h
        overlap_ratio = overlap_area / (area[idxs[:last]] + area[i] - overlap_area)

        if softnms == "gaussian":
            weight = np.exp(-overlap_ratio * overlap_ratio / sigma)
        elif softnms == "linear":
            weight = 1 - overlap_ratio
        elif softnms is None:
            weight = overlap_ratio < iou_thresh
            weight = weight.astype('int')
        else:
            raise ValueError("nms method can only be: gaussian, linear, None")

        current_confidence = current_confidence * weight # 为了使用nms才采用weight这种方式
        need_to_be_deleted_idx = np.concatenate(([last], np.where(current_confidence < conf_thresh)[0]))
        # need_to_be_deleted_idx = np.concatenate(([last], np.where(overlap_ratio > iou_thresh)[0]))
        idxs = np.delete(idxs, need_to_be_deleted_idx)

    # if the number of final bboxes is less than keep_top_k, we need to pad it.
    # TODO
    return conf_keep_idx[pick]
