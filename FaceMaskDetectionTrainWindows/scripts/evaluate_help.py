# -*- encoding=utf-8 -*-

'''
基本思路：
    1. 从标签文件中读取ground truth，读取格式为：<img_name> <class_name> <xmin> <ymin> <xmax> <ymax> <difficult_flag>
    2. 从检测结果中读取predict result，读取格式为：<img_name> <class_name> <confidence> <xmin> <ymin> <xmax> <ymax>
    3. 为了避免ground truth和predict result中图片数量不一致，需要对ground truth和predict result做一次校准，主要是刘林赟跑测试的时候，
       测试集的图片数量要少于训练集，所以需要将ground truth中多余的图片去除，不去除的话，会当做漏检，影响结果。当然，去除了，或许也会影响结果。。
       当然，这只是一些具体业务相关的，最好的就是，ground truth和 predict result中图片数量是一样的。这样结果完全客观公正。
'''

import os
from data_process.pascal_loader import load_pascal_data
from evaluation.evaluate_utils import evaluate
from tqdm import tqdm
import numpy as np
from copy import copy
import argparse
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", help="Path to your config path.", type=str)
parser.add_argument("--img_path", help="Path to the test image path", type=str)

args = parser.parse_args()

config_path = args.config_path
img_path = args.img_path

valset_ground_truth_dir = '/home/daniel/Data/HeadFaceBodyData/ReLabelManually/Test/images/'
predict_result_file_path = '/home/daniel/Data/迅雷下载/det_rst.txt'

name2id = {'body':0, 'head':1, 'face':2}  # 这个决定了会加载ground truth的哪些类别

id2name = {0:'body', 1:'head', 2:'face'}

img_val, bbox_val, label_val, difficult_val, image_names = load_pascal_data(valset_ground_truth_dir,
                                                                           class2id=name2id,
                                                                           bboxes_normalized=False,
                                                                           return_image_name=True,
                                                                           downsample_image=False
                                                                           )
gt_info = []
predict_info = []
id2name = {0:'body', 1:'head', 2:'face'}
# 加载ground truth
print("Start to load ground truth.")
for i in tqdm(range(len(img_val[:]))):
    img_name = image_names[i]
    img = np.copy(img_val[i])

    for j in range(len(bbox_val[i])):
        class_name = id2name[label_val[i][j]]  # 因为读取的gt_label其实是数字id，这里需要再转回到text 累呗
        xmin, ymin, xmax, ymax = bbox_val[i][j]
        difficult_flag = difficult_val[i][j]
        gt_info.append([img_name, class_name, xmin, ymin, xmax, ymax, difficult_flag])


# 加载predict result
print("Start to load predict result.")
with open(predict_result_file_path) as f:
    for line in f.readlines():
        line = line.strip().split(" ")
        predict_info.append(line)


# 从ground truth中过滤掉predict info中没有的图片，主要是面对刘林赟的需求，如果两者图片数量一致，不需要做这一步

gt_info_bk = copy(gt_info)
gt_info = []
predict_image_names = list(set(np.array(predict_info)[:, 0]))
for line in gt_info_bk:
    if line[0] in predict_image_names :
        gt_info.append(line)


print("Start to evaluate.")
evaluate_result = evaluate(gt_info, predict_info, use_07_metric=False, omit_difficult=True)

fig, ax = plt.subplots()
ax.xaxis.set_major_locator(MultipleLocator(0.1))
ax.yaxis.set_major_locator(MultipleLocator(0.1))
ax.xaxis.set_minor_locator(AutoMinorLocator(0.05))
ax.xaxis.set_minor_locator(AutoMinorLocator(0.05))
ax.grid(which='both')

ax.set_xlim(0, 1)
ax.set_ylim(0, 1.05)

title_text = ""
for cate in evaluate_result.keys():
    ap = evaluate_result[cate]['ap']
    precision =evaluate_result[cate]['precision']
    recall = evaluate_result[cate]['recall']
    plt.plot(recall, precision, label=cate)
    title_text += "%s : %.3f  " % (cate, ap)

# head_ap = evaluate_result['head']['ap']
# head_precision =evaluate_result['head']['precision']
# head_recall = evaluate_result['head']['recall']
# plt.plot(head_recall, head_precision, label='head')
#
# face_ap = evaluate_result['face']['ap']
# face_precision =evaluate_result['face']['precision']
# face_recall = evaluate_result['face']['recall']
# plt.plot(face_recall, face_precision, label='face')


plt.xlabel("recall")
plt.ylabel("precision")
plt.title(title_text)

ax.legend()
plt.show()



