# -*- coding:utf-8 -*-
import cv2
import os
import yaml
import time
import argparse
import numpy as np
from PIL import Image
from keras.models import load_model

from anchor.anchor_generator import generate_anchors
from anchor.anchor_decode import decode_bbox
from loss.ssd_loss import loc_loss, cls_loss
from inference.nms import single_class_non_max_suppression
from utils.utils import generate_anchor_config



def init_config(model, config_path):
    global anchors, anchors_exp, class2id, id2class
    # 构建模型

    if not os.path.exists(config_path):
        raise ValueError("Your config path is not exist.")

    with open(config_path) as f:
        config = yaml.load(f)

    detection_layer_config = config['detection_layer_config']

    anchor_config = generate_anchor_config(model, detection_layer_config)


    anchors = generate_anchors(anchor_config['feature_map_sizes'],
                               anchor_config['anchor_scales'],
                               anchor_config['anchor_ratios'])

    anchors_exp = np.expand_dims(anchors, axis=0)

    class2id = config['class2id']
    id2class = {}
    for k, v in class2id.items():
        id2class[v] = k

    # global anchors, anchors_exp, class2id, id2class
    # id2class = {0:'Electric bike', 1:'Bicycle'}

def sample_inference(img,
                     model,
                     conf_thresh=0.5,
                     iou_thresh=0.5,
                     target_shape=(160,160),
                     draw_result=True,
                     show_result=False,
                     softnms=None,
                     sigma = 0.1
                     ):
    image = np.copy(img)
    output_info = []
    height, width, _ = image.shape
    image_resized = cv2.resize(image, target_shape)
    image_np = image_resized / 255.0
    image_exp = np.expand_dims(image_np, axis=0)
    loc_branch, cls_branch = model.predict(image_exp)
    y_bboxes = decode_bbox(anchors_exp, loc_branch)[0]
    y_cls = np.squeeze(cls_branch, axis=0)

    # 对每个类单独进行nms
    for class_id in range(y_cls.shape[1]):
        keep_idxs = single_class_non_max_suppression(y_bboxes,
                                                      y_cls[:, class_id],
                                                      conf_thresh=conf_thresh,
                                                      iou_thresh=iou_thresh,
                                                      softnms = softnms,
                                                      sigma=sigma
                                                      )

        for idx in keep_idxs:
            conf = float(y_cls[idx, class_id])
            bbox = y_bboxes[idx]
            # 裁剪坐标，将超出图像边界的坐标裁剪
            xmin = max(0, int(bbox[0] * width))
            ymin = max(0, int(bbox[1] * height))
            xmax = min(int(bbox[2] * width), width)
            ymax = min(int(bbox[3] * height), height)

            # xmin = int(bbox[0] * width)
            # ymin = int(bbox[1] * height)
            # xmax = int(bbox[2] * width)
            # ymax = int(bbox[3] * height)
            if draw_result:
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (80, 255, 2), 2)
                # cv2.circle(image, ((xmin + xmax) /2, (ymin + ymax)/2), 10, (6,255,8),3)
                cv2.line(image,(xmin, ymin), (xmax, ymax), (80, 127, 255), 2)
                cv2.putText(image, "%s: %.2f" % ( id2class[class_id], conf), (xmin + 2, ymin - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(80, 17, 255))
            output_info.append([class_id, conf, xmin, ymin, xmax, ymax])

    if show_result:
        Image.fromarray(image[:, :, ::-1]).show()
    return output_info



def run_on_video(video_path, output_video_name, conf_thresh):
    cap = cv2.VideoCapture(video_path)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter(output_video_name, fourcc, int(fps), (int(width), int(height)))
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    if not cap.isOpened():
        raise ValueError("Video open failed.")
        return
    status = True
    idx = 0
    while status:
        start_stamp = time.time()
        status, img_raw = cap.read()
        read_frame_stamp = time.time()
        if (status):
            sample_inference(img_raw,
                             conf_thresh,
                             iou_thresh=0.5,
                             target_shape=(352, 200),
                             draw_result=True,
                             show_result=False)

            inference_stamp = time.time()
            writer.write(img_raw)
            write_frame_stamp = time.time()
            idx += 1
            print("%d of %d" % (idx, total_frames))
            print("read_frame:%f, infer time:%f, write time:%f" % (read_frame_stamp - start_stamp,
                                                                   inference_stamp - read_frame_stamp,
                                                                   write_frame_stamp - inference_stamp))
    writer.release()
