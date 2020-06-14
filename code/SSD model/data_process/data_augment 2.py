# -*- coding:utf-8 -*-
import numpy as np
import cv2
from PIL import Image, ImageEnhance
from random import shuffle, choice
import random

def image_flip_horizontal(image, bboxes, labels):
    '''
    图片左右翻转，注意：图片翻转以后，bounding box坐标也要左右调整
    :param image:
    :param bboxes:
    :param labels:
    :return:
    '''
    image_fliped = image[:, ::-1, :]
    bboxes_fliped = [[1-bbox[2], bbox[1], 1-bbox[0], bbox[3]] for bbox in bboxes]
    return image_fliped, bboxes_fliped, labels

def cal_ioa(bboxes_a, bbox_b):
    '''
    计算IoA的大小，坐标顺序[xmin, ymin, xmax, ymax]
    :param bboxes_a: 一组bbox坐标列表，
    :param bbox_b: 一个bbox坐标
    :return:
    '''
    bboxes_a = np.array(bboxes_a)
    bbox_b = np.array(bbox_b)
    union_xmin = np.maximum(bboxes_a[:, 0], bbox_b[0])
    union_ymin = np.maximum(bboxes_a[:, 1], bbox_b[1])
    union_xmax = np.minimum(bboxes_a[:, 2], bbox_b[2])
    union_ymax = np.minimum(bboxes_a[:, 3], bbox_b[3])
    union_width = np.maximum(union_xmax - union_xmin, 0)
    union_height = np.maximum(union_ymax - union_ymin, 0)
    union_area = union_width * union_height
    bboxes_a_area = (bboxes_a[:, 2] - bboxes_a[:, 0]) * (bboxes_a[:, 3] - bboxes_a[:, 1]) + 1e-6
    return union_area / bboxes_a_area


def image_random_crop(image, bboxes, labels, min_width_ratio, min_height_ratio):
    '''
    随机crop图像，注意：图像crop以后，坐标也要跟随crop一并重新调整，别裁掉的目标物体的bounding box和label也要删除
    :param image:
    :param bboxes:
    :param labels:
    :param min_width_ratio:
    :param min_height_ratio:
    :return:
    '''
    croped_width_ratio = np.random.random() * (1 - min_width_ratio) + min_width_ratio # 随机生成新图形的宽
    croped_height_ratio = np.random.random() * (1 - min_height_ratio) + min_height_ratio

    croped_begin_x = np.random.random() * (1 - croped_width_ratio)
    croped_begin_y = np.random.random() * (1 - croped_height_ratio)

    src_img_height, src_img_width, _ = image.shape
    croped_begin_x_reversed = int(croped_begin_x * src_img_width)
    croped_begin_y_reversed = int(croped_begin_y * src_img_height)
    croped_end_x_reversed = int((croped_begin_x + croped_width_ratio) * src_img_width)
    croped_end_y_reversed = int((croped_begin_y + croped_height_ratio) * src_img_height)

    croped_image = image[croped_begin_y_reversed: croped_end_y_reversed,
                         croped_begin_x_reversed: croped_end_x_reversed, :]

    if croped_image.size == 0:
        print("Zero size image occur.!!!!!!!!!!!!!!!!")

    if len(bboxes) == 0:  # 如果没有bbox信息，说明可能是图像没有bbox信息，或者这是图像分类应用， 直接返回
        return croped_image, [], []

    croped_bbox = [croped_begin_x, croped_begin_y, croped_begin_x + croped_width_ratio,
                   croped_begin_y + croped_height_ratio]

    ioa = cal_ioa(bboxes, croped_bbox)
    bboxes = np.array(bboxes)
    labels = np.array(labels)
    alive_bboxes = bboxes[ioa > 0.2]
    alive_labels = labels[ioa > 0.2]
    # 如果一经过裁剪， 没有目标物了，就直接返回
    if alive_bboxes.size == 0:
        return croped_image, [], []

    xmin = np.maximum(alive_bboxes[:, 0:1], croped_bbox[0])
    ymin = np.maximum(alive_bboxes[:, 1:2], croped_bbox[1])
    xmax = np.minimum(alive_bboxes[:, 2:3], croped_bbox[2])
    ymax = np.minimum(alive_bboxes[:, 3:4], croped_bbox[3])

    alive_bboxes_new = np.concatenate((xmin, ymin, xmax, ymax), axis=-1)
    alive_bboxes_adjusted = (alive_bboxes_new - np.array([croped_begin_x, croped_begin_y,croped_begin_x, croped_begin_y])) / \
                            np.array([croped_width_ratio, croped_height_ratio, croped_width_ratio,
                                      croped_height_ratio])

    return croped_image, alive_bboxes_adjusted, alive_labels


def random_color_adjust(image, adjust_scale):
    '''
    使用PIL的ImageEnhance模块对每个图片进行随机的亮度、颜色、对比度、锐利度增强
    :param image:
    :param adjust_scale:
    :return:
    '''
    def random_number():
        return 2 * adjust_scale * np.random.random() + 1 - adjust_scale

    adjustors = [ImageEnhance.Brightness, ImageEnhance.Color, ImageEnhance.Contrast, ImageEnhance.Sharpness]
    shuffle(adjustors) # 随机生成色彩调整器的顺序

    pil_img = Image.fromarray(image)

    for adjustor in adjustors:
        pil_img = adjustor(pil_img).enhance(random_number())
    return np.array(pil_img)

def attach_object_fuc_help(image, attach_img, xmin, ymin, xmax, ymax):
    attach_img_height, attach_img_width, _ = attach_img.shape
    if attach_img_width < (xmax - xmin):
        xmax = xmin + attach_img_width
    if attach_img_height < (ymax - ymin):
        ymax = ymin + attach_img_height
        
    attach_img_xstart = int((attach_img_width - (xmax - xmin)) * random.random())
    attach_img_ystart = int((attach_img_height - (ymax - ymin)) * random.random())
    crop_img = attach_img[attach_img_ystart: attach_img_ystart + (ymax - ymin),
                            attach_img_xstart: attach_img_xstart + (xmax - xmin)]

    mask = (crop_img == 0).astype(np.int8)
    image[ymin:ymax, xmin:xmax, :] = (image[ymin:ymax, xmin:xmax, :] * mask).astype(np.uint8) + crop_img
    return image


def attach_object_fuc(image, bboxes, labels, attach_object_imgs):
    imgHeight, imgWidth, _ = image.shape
    attach_img = random.choice(attach_object_imgs)
    for idx, label in enumerate(labels):
        if label == 1:
            bbox = bboxes[idx]
            xmin = int(bbox[0] * imgWidth)
            ymin = int(bbox[1] * imgHeight)
            xmax = int(bbox[2] * imgWidth)
            ymax = int(bbox[3] * imgHeight)
            height = ymax - ymin
            width = xmax - xmin
            ystart = ymin + height * (0.6 + 0.2 * random.random())
            yend = ymin + height * (0.9 + random.random() * 0.3)
            if yend > imgHeight:
                yend = imgHeight
            xstart = xmin + width * (0.2 + random.random() * 0.6)
            xend = xstart + width * (random.random()*0.8 + 0.2)
            xendMax = min(xmax + width * random.random() * 0.1, imgWidth)
            if xend > xendMax:
                xend = xendMax
            # print (xstart, ystart, xend,yend)
            image = attach_object_fuc_help(image, attach_img, int(xstart), int(ystart), int(xend), int(yend))
    return image



def augment(image,
            bboxes,
            labels,
            target_size=(200, 200),
            min_width_ratio=0.7,
            min_height_ratio=0.8,
            random_horizontal_flip=True,
            random_crop=True,
            color_aug=True,
            color_aug_scale=0.5,
            attach_object=False,
            attach_object_prob=0,
            attach_object_imgs = []
            ):
    '''
    数据增强主函数
    :param image: numpy 3d array.
    :param bboxes: list of bounding box.
    :param labels: list of labels.
    :param target_size: 输出图像大小
    :param min_width_ratio: 在crop时候，crop后至少保持的宽度范围，例如0.7意味这裁剪后的宽度，不能低于原来宽度的70%
    :param min_height_ratio: 同上
    :param random_horizontal_flip: 随机水平翻转
    :param random_crop: 随机crop图像
    :param color_aug: 颜色增强
    :param color_aug_scale: 颜色增强的尺度
    :return:
    '''
    if attach_object and random.random() < attach_object_prob:
        image = attach_object_fuc(image, bboxes, labels, attach_object_imgs)

    if random_crop:
        image, bboxes, labels = image_random_crop(image, bboxes, labels, min_width_ratio, min_height_ratio)
    if random_horizontal_flip and np.random.random() > 0.5:
        image, bboxes, labels = image_flip_horizontal(image, bboxes, labels)
    image = cv2.resize(image, (target_size[0], target_size[1]))
    if color_aug:
        image = random_color_adjust(image, adjust_scale=color_aug_scale)
    return image, bboxes, labels

