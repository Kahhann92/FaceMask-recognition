# -*- encoding:utf-8 -*-
import os
import cv2
import signal
import numpy as np
from copy import copy
from multiprocessing import Pool
from functools import partial
from anchor.anchor_generator import generate_anchors
from data_process.data_augment import augment
from anchor.anchor_encode import encode_bboxes_to_anchor

# 中断信号
def init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def min_length_resize(image, min_length = 400):
    '''
    resize图像到一个小尺度，避免将1920*1080这种大尺度的图片加载到内存中
    :param image: ndarray类型的图像
    :param min_length: 保持短边至少达到min_length以上
    :return:
    '''
    height, width, _ = image.shape
    if height > width and width> min_length:
        new_width = min_length
        new_height = int(float(min_length) / width * height)
        return cv2.resize(image, (new_width, new_height))
    elif height <= width and height > min_length:
        new_height = min_length
        new_width = int(float(min_length) / height * width)
        return cv2.resize(image, (new_width, new_height))
    else:
        return image


def load_attach_imgs(imgDir):
    attachs = []
    for imgName in os.listdir(imgDir):
        imgPath = os.path.join(imgDir, imgName)
        img = cv2.imread(imgPath)
        img = min_length_resize(img, 200)
        attachs.append(img)
    return attachs

attach_dir = 'D:/Projects/FaceMaskDetectionTrainWindows/attach_data/mask'
attach_object_imgs = load_attach_imgs(attach_dir)

def image_process(data, train_mode, preprocess_config,  target_size, anchors, num_class, omit_difficult=True):
    sample_image = data[0]
    sample_bboxes = data[1]
    sample_labels = data[2]
    sample_difficulties = data[3]

    # 获取数据增强配置信息
    if 'color_aug' in preprocess_config.keys():
        color_aug = preprocess_config['color_aug']['open']
        color_aug_scale = preprocess_config['color_aug']['scale']
    else:
        color_aug = False
        color_aug_scale = 0

    if 'random_crop' in preprocess_config.keys():
        random_crop = preprocess_config['random_crop']['open']
        min_width_ratio = preprocess_config['random_crop']['min_width_ratio']
        min_height_ratio = preprocess_config['random_crop']['min_height_ratio']
    else:
        random_crop = False
        min_width_ratio = 1.0
        min_height_ratio = 1.0

    if 'random_horizontal_flip' in preprocess_config.keys():
        random_horizontal_flip = preprocess_config['random_horizontal_flip']['open']
    else:
        random_horizontal_flip = False

    if 'attach_object' in preprocess_config.keys():
        attach_object = preprocess_config['attach_object']['open']
        attach_object_prob = preprocess_config['attach_object']['prob']

    else:
        attach_object = False
        attach_object_prob = 0
    # min_width_ratio


    # 过滤掉困难实例
    if omit_difficult and len(sample_difficulties) > 0 and len(sample_bboxes) == len(sample_difficulties):
        sample_bboxes_cp = copy(sample_bboxes)
        sample_labels_cp = copy(sample_labels)
        # sample_difficulties_np = copy(sample_difficulties)
        sample_bboxes = []
        sample_labels = []
        # sample_difficulties = []
        for i in range(len(sample_difficulties)):
            if not int(sample_difficulties[i]):
                sample_bboxes.append(sample_bboxes_cp[i])
                sample_labels.append(sample_labels_cp[i])

    if train_mode:
        # sample_image = cv2.resize(sample_image, target_size)
        sample_image, sample_bboxes, sample_labels = augment(sample_image,
                                                              sample_bboxes,
                                                              sample_labels,
                                                              target_size=target_size,
                                                              min_width_ratio=min_width_ratio,
                                                              min_height_ratio=min_height_ratio,
                                                              random_horizontal_flip=random_horizontal_flip,
                                                              random_crop=random_crop,
                                                              color_aug=color_aug,
                                                              color_aug_scale=color_aug_scale,
                                                              attach_object=attach_object,
                                                              attach_object_prob=attach_object_prob,
                                                             attach_object_imgs=attach_object_imgs
                                                             )

    else:
        sample_image = cv2.resize(sample_image, (target_size[0], target_size[1]))

    # 如果没有bboxes了，就直接返回
    if len(sample_bboxes) == 0:
        encoded_result = np.zeros((anchors.shape[0], 4 + num_class))
        return sample_image, encoded_result,

    encoded_result = encode_bboxes_to_anchor(
                            sample_bboxes, sample_labels, anchors, num_class, iou_thresh=0.35)
    return sample_image, encoded_result


def data_generator_multiprocess(images,
                                bboxes,
                                labels,
                                difficulties,
                                train_mode,
                                anchor_config,
                                preprocess_config,
                                input_shape,
                                num_class,
                                batch_size=32,
                                pool_nb=32):
    '''
    目标检测算法的数据生成器
    :param images: list of image ndarray.
    :param bboxes: list of list, 存放每个图片的bounding box
    :param labels: list of list, 存放每个图片的labels
    :param difficulties: list of list, 存放每个图片的bounding box是否属于difficult
    :param train_mode: 是训练模式还是评估模式
    :param anchor_config: anchor配置
    :param preprocess_config: 预训练参数配置
    :param input_shape: 检测网络的输入大小
    :param num_class: 如果是用sigmoid激活，num_class为总类别数，如果是softmax激活，则是总类别数+1（+1为背景类）.
    :param batch_size:
    :param pool_nb: 数据预处理用到了python的多线程，batch_size是pool_nb的1倍或者整数倍即可
    :return:
    '''
    # 生成anchors
    anchors = generate_anchors(anchor_config['feature_map_sizes'],
                               anchor_config['anchor_scales'],
                               anchor_config['anchor_ratios'])

    image_process_train = partial(image_process,
                                  train_mode=True,
                                  preprocess_config=preprocess_config,
                                  target_size=input_shape,
                                  anchors=anchors,
                                  num_class = num_class,
                                  omit_difficult=False)
    image_process_test = partial(image_process,
                                 train_mode=False,
                                 preprocess_config=preprocess_config,
                                 target_size=input_shape,
                                 anchors=anchors,
                                 num_class = num_class,
                                 omit_difficult=False)

    num_samples = len(images)
    random_index = np.random.permutation(num_samples)
    pool = Pool(pool_nb, init_worker)
    idx = 0
    # 在while循环中生成数据，使用yield迭代
    while True:
        batch_images = []
        batch_bboxes = []
        batch_labels = []
        batch_difficulties = []

        if idx >= num_samples - 1:
            idx = 0
            random_index = np.random.permutation(num_samples)
        endpoint = min(idx + batch_size, num_samples)

        # print("This batch length: %d" % (endpoint - idx))
        for i in range(idx, endpoint):
            batch_images.append(images[random_index[i]])
            batch_bboxes.append(bboxes[random_index[i]])
            batch_labels.append(labels[random_index[i]])
            batch_difficulties.append(difficulties[random_index[i]])

        batch_data = zip(batch_images, batch_bboxes, batch_labels, batch_difficulties)

        try:
            if train_mode:
                processed_data = pool.map(image_process_train, batch_data)
            else:
                processed_data = pool.map(image_process_test, batch_data)
        except KeyboardInterrupt:
            print("Keyboard Interrupt detected, terminating the pool...")
            pool.terminate()
            pool.join()
        except ValueError:
            continue

        batch_processed_images = []
        batch_encoded_results = []

        for k in range(len(processed_data)):
            batch_processed_images.append(processed_data[k][0])
            batch_encoded_results.append(processed_data[k][1])

        idx += batch_size

        batch_encoded_results = np.array(batch_encoded_results)

        yield np.array(batch_processed_images, dtype=np.float32) / 255.0,\
              [batch_encoded_results, batch_encoded_results[:, :, 4:]]
