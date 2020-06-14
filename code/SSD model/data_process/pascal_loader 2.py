# -*- encoding=utf-8 -*-
import os
import cv2
from random import shuffle
import numpy as np
from lxml import etree
from tqdm import tqdm

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


def recursive_parse_xml_to_dict(xml):
    """Recursively parses XML contents to python dict.
    We assume that `object` tags are the only ones that can appear
    multiple times at the same level of a tree.
    Args:
      xml: xml tree obtained by parsing XML file contents using lxml.etree
    Returns:
      Python dictionary holding XML contents.
    """
    if not xml:
        return {xml.tag: xml.text}
    result = {}
    for child in xml:
        child_result = recursive_parse_xml_to_dict(child)
        if child.tag != 'object':
            result[child.tag] = child_result[child.tag]
        else:
            if child.tag not in result:
                result[child.tag] = []
            result[child.tag].append(child_result[child.tag])
    return {xml.tag: result}


def read_xml(xml_path):
    with open(xml_path) as f:
        xml_str = f.read()
    xml = etree.fromstring(xml_str)
    data = recursive_parse_xml_to_dict(xml)
    return data


def load_coords(xml_path, image_height, image_width, class2id, bboxes_normalized=True, image_true_height=0, image_true_width=0):
    '''
    从xml中读取图像标注信息
    :param xml_path:
    :param image_height:
    :param image_width:
    :param class2id:
    :param bboxes_normalized:
    :param image_true_height:
    :param image_true_width:
    :return:
    '''
    data = read_xml(xml_path)
    bboxes = []
    labels = []
    difficulties = []
    if 'object' in data['annotation']:
        for obj in data['annotation']['object']:
            if obj['name'] in class2id.keys():
                difficult_flag = int(float(obj['difficult']))
                if bboxes_normalized:
                    xmin, ymin, xmax, ymax = float(obj['bndbox']['xmin']) / image_width, float(obj['bndbox']['ymin']) / image_height,\
                                         float(obj['bndbox']['xmax']) / image_width, float(obj['bndbox']['ymax']) / image_height

                else:
                    xmin, ymin, xmax, ymax = float(obj['bndbox']['xmin'])/image_width * image_true_width,\
                                             float(obj['bndbox']['ymin'])/image_height * image_true_height, \
                                             float(obj['bndbox']['xmax'])/image_width * image_true_width,\
                                             float(obj['bndbox']['ymax'])/image_height * image_true_height
                # 需要这么做的原因是重庆那边数据标注，有的是左上-右下，有的是左下-右上
                if xmin > xmax:
                    xmin, xmax = xmax, xmin
                if ymin > ymax:
                    ymin, ymax = ymax, ymin

                bboxes.append([xmin, ymin, xmax, ymax])
                labels.append(class2id[obj['name']])
                difficulties.append(difficult_flag)
    else:
        print(xml_path)
    return bboxes, labels, difficulties


def load_pascal_data(root_path, class2id, bboxes_normalized=True, return_image_name=False, downsample_image=False):
    '''
    读取pascal voc格式的数据
    :param root_path: 数据的目录，注意文件夹里面的内容应该是jpg、jpeg，png格式的图像数据，以及xml格式的标注
    :param class2id: dict，class name与id的转换，注意，这里只加载class2id的keys里面的类别，例如，pascal voc虽然有20类，如果我们只想加载
                     person和dog两类，则{'person':0, 'dog':1}这样子就可以了，其他类别不会加载
    :param bboxes_normalized: 是否把bounding box归一化到0~1
    :param return_image_name: 是否将图片名字一并返回，对于评估模型的时候，需要根据图片名字匹配，则需要这个图片名字。
    :param downsample_image: 是否需要将图片下采样，如过原图像比较大，而且需要将图片加载在内容，对图片resize到一个小尺度是一个最好的选择
    :return:
    '''
    all_image_files = [os.path.join(root_path, x)
                       for x in os.listdir(root_path)
                       if x.endswith('jpg') or x.endswith('jpeg') or x.endswith('png')]
    shuffle(all_image_files) # 打乱图片顺序，然后可以在后面切分训练集和测试集
    image_arrs, bboxes, labels, difficulties, image_names = [], [], [], [], []

    for idx, image_path in tqdm(enumerate(all_image_files[:])):
        if (idx > 200):
            continue
        # print("Without", image_path)
        # print("utf8", image_path.encode("utf-8"))
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        image_names.append(os.path.split(image_path)[1])
        height, width, _ = image.shape  # 这个原坐标很重要
        if downsample_image:
            image = min_length_resize(image, 400)
        image_true_height, image_true_width, _ = image.shape

        xml_file_path = os.path.splitext(image_path)[0] + ".xml"
        if os.path.exists(xml_file_path):
            sample_bboxes, sample_labels, sample_difficulties = load_coords(xml_file_path,
                                                                            height, width,
                                                                            class2id,
                                                                            bboxes_normalized,
                                                                            image_true_height,
                                                                            image_true_width)
            # print(xml_file_path, sample_bboxes, sample_labels)
            image_arrs.append(image)
            bboxes.append(sample_bboxes)
            labels.append(sample_labels)
            difficulties.append(sample_difficulties)
    if return_image_name:
        return image_arrs, bboxes, labels, difficulties, image_names
    return image_arrs, bboxes, labels, difficulties
