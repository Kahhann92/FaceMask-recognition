import cv2
from PIL import Image
import numpy as np

def show_matched_anchor(batch_images, batch_labels, anchors, i):
    img = batch_images[i]
    height, width, _ = img.shape
    labels = batch_labels[i]
    img = (img * 255).astype('uint8')
    for j in range(len(labels)):
        label = labels[j].sum()
        if label:
            cv2.rectangle(img, (int(anchors[j, 0] * width), int(anchors[j,1] * height)),
                          (int(anchors[j, 2] * width), int(anchors[j, 3] * height)), (0,0, 255), 1 )
    Image.fromarray(img).show()

def show_src_data(imgs, all_bboxes, i):
    img = imgs[i]
    bboxes = all_bboxes[i]
    height, width, _ = img.shape
    for bbox in bboxes:
        cv2.rectangle(img, (int(bbox[0] * width), int(bbox[1] * height)),
                      (int(bbox[2] * width), int(bbox[3] * height)), (0, 0, 255), 1)
    Image.fromarray(img).show()

def show_anchors(anchors, img, i,):
    img = (img * 255).astype(np.uint8)
    height, width, _ = img.shape
    for bbox in  anchors[i:i+ 4]:
        cv2.rectangle(img, (int(bbox[0] * width), int(bbox[1] * height)),
                      (int(bbox[2] * width), int(bbox[3] * height)), (0, 0, 255), 1)
    Image.fromarray(img).show()