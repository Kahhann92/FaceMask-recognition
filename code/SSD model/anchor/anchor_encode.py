# -*- coding:utf-8 -*-
import numpy as np

def find_match_anchors(bbox, anchors, iou_thresh=0.5):
    '''
    根据GT找到与之最匹配的anchors，也就是anchors与GT的IOU大于阈值，即认为这些anchors是正样本
    :param bbox: 1D array, [xmin, ymin, xmax, ymax]
    :param anchors: 2D array, N x 4
    :return:
    '''
    inter_xmin = np.maximum(bbox[0], anchors[:, 0])
    inter_ymin = np.maximum(bbox[1], anchors[:, 1])
    inter_xmax = np.minimum(bbox[2], anchors[:, 2])
    inter_ymax = np.minimum(bbox[3], anchors[:, 3])
    inter_width = np.maximum(0, inter_xmax -inter_xmin)
    inter_height = np.maximum(0, inter_ymax - inter_ymin)
    inter_area = inter_width * inter_height
    bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    anchor_areas = (anchors[:, 2] - anchors[:, 0]) * (anchors[:, 3] - anchors[:, 1])
    union_area = anchor_areas - inter_area + bbox_area
    iou = inter_area / union_area
    match_idx = np.where(iou > iou_thresh)[0]
    if len(match_idx) == 0: # 如果没有与GT的IOU大于阈值的的anchor，则取IOU最大的一个anchor作为匹配的正样本
        match_idx = np.argsort(iou)[-1:]

    match_iou = iou[match_idx]
    return match_idx, match_iou


def encode_one_item_with_anchor(bbox, anchors, variance = [0.1, 0.1, 0.2, 0.2] ):
    '''
    # 将一个GT bounding box编码到与之匹配的anchor上
    :param head_face_bbox: a list of head or face bbox
    :param anchors: matched anchors, 2D arrays, N x 4
    :return:
    '''
    bbox_cx = (bbox[0] + bbox[2]) / 2.0
    bbox_cy = (bbox[1] + bbox[3]) /  2.0
    bbox_w = np.maximum(bbox[2] - bbox[0],  1e-7)
    bbox_h = np.maximum(bbox[3] - bbox[1], 1e-7)
    anchor_cx = (anchors[:, 0:1] + anchors[:, 2:3]) / 2
    anchor_cy = (anchors[:, 1:2] + anchors[:, 3:]) / 2
    anchor_w = anchors[:, 2:3] - anchors[:, 0:1]
    anchor_h = anchors[:, 3:] - anchors[:, 1:2]
    encoded_cx = (bbox_cx - anchor_cx) / anchor_w
    encoded_cy = (bbox_cy - anchor_cy) / anchor_h
    encoded_w = np.log(bbox_w / anchor_w)
    encoded_h = np.log(bbox_h / anchor_h)
    encoded_bbox = np.concatenate([encoded_cx, encoded_cy, encoded_w, encoded_h], axis=-1)
    return encoded_bbox / np.array(variance)


# def iou(multi_bboxes, bbox):
#     inter_xmin = np.maximum(multi_bboxes[:, 0], bbox[0])
#     inter_ymin = np.maximum(multi_bboxes[:, 1], bbox[1])
#     inter_xmax = np.minimum(multi_bboxes[:, 2], bbox[2])
#     inter_ymax = np.minimum(multi_bboxes[:, 3], bbox[3])
#     inter_width = np.maximum(0, inter_xmax -inter_xmin)
#     inter_height = np.maximum(0, inter_ymax - inter_ymin)
#
#     inter_area = inter_width * inter_height
#     bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
#
#     multi_areas = (multi_bboxes[:, 2] - multi_bboxes[:, 0]) * (multi_bboxes[:, 3] - multi_bboxes[:, 1])
#     union_area = multi_areas - inter_area + bbox_area
#
#     iou_value = inter_area / union_area
#     return iou_value


def encode_bboxes_to_anchor(bboxes, labels, anchors, num_class, iou_thresh=0.35):
    '''
    将bboxes编码到anchors上面
    :param bboxes: 图片的ground truth bboxes
    :param labels: 图片的ground truth labels
    :param anchors:
    :param num_class: 要注意使用的softmax还是sigmoid激活函数，如果用softmax，要多一个背景类
    :param iou_thresh: iou匹配阈值
    :return:
    '''
    encoded_result = np.zeros((anchors.shape[0], 4 + num_class)) # 这里需要注意使用的是
    for idx, bbox in enumerate(bboxes):
        label = labels[idx]
        match_idx, match_iou = find_match_anchors(bbox, anchors, iou_thresh)
        encoded_bbox = encode_one_item_with_anchor(bbox, anchors[match_idx])
        encoded_result[match_idx, :4] = encoded_bbox
        encoded_result[match_idx, 4 + label] = 1 # 省去 one_hot 步骤， 直接在相应类别编号位置上添加标注
    return encoded_result
