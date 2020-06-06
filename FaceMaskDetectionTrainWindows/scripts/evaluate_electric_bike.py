import os
from data_process.pascal_loader import load_pascal_data
from inference.infer import sample_inference
from evaluation.evaluate_utils import evaluate

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator

valset_path = '/home/daniel/Data/CWDataset/Ebike/valset'

img_val, bbox_val, label_val, difficult_val = load_pascal_data(valset_path,
                                                               class2id={'ebike':0, 'bike':1},
                                                               bboxes_normalized=False
                                                               )


gt_info = []
predict_info = []

'''
    1. 标签按照以下格式存放为一个文件，读取为列表，每个列表为一行ground truth
    <img_name> <class_name> <xmin> <ymin> <xmax> <ymax> <difficult_flag>

    2. 预测结果按照以下格式存放，读取为列表，每个列表为一行ground truth
    <img_name> <class_name> <confidence> <xmin> <ymin> <xmax> <ymax>
'''

id2name = {0:"Electric bike",1:'Bike'}

for i in range(len(img_val[:])):
    img_name = str(i) + ".jpg"
    for j in range(len(bbox_val[i])):
        class_name = id2name[label_val[i][j]]
        xmin, ymin, xmax, ymax = bbox_val[i][j]
        difficult_flag = difficult_val[i][j]
        gt_info.append([img_name, class_name, xmin, ymin, xmax, ymax, difficult_flag])

    predict_result = sample_inference(img_val[i],
                                     conf_thresh=0.01,
                                     iou_thresh=0.5,
                                     target_shape=(160,160),
                                     draw_result=False,
                                     show_result=False,
                                     softnms=None,
                                     sigma = 0.1
                                     )
    for k in range(len(predict_result)):
        class_name = id2name[predict_result[k][0]]
        conf, xmin, ymin, xmax, ymax = predict_result[k][1:]
        predict_info.append([img_name, class_name, conf, xmin, ymin, xmax, ymax])

evaluate_result = evaluate(gt_info, predict_info)

fig, ax = plt.subplots()
ax.xaxis.set_major_locator(MultipleLocator(0.1))
ax.yaxis.set_major_locator(MultipleLocator(0.1))
ax.xaxis.set_minor_locator(AutoMinorLocator(0.05))
ax.xaxis.set_minor_locator(AutoMinorLocator(0.05))
ax.grid(which='both')

ax.set_xlim(0, 1)
ax.set_ylim(0, 1.05)

Bike_ap = evaluate_result['Bike']['ap']
Boke_precision =evaluate_result['Bike']['precision']
Boke_recall = evaluate_result['Bike']['recall']
plt.plot(Boke_recall, Boke_precision, label='Bike')

Electric_bike_ap = evaluate_result['Electric bike']['ap']
Electric_bike_precision =evaluate_result['Electric bike']['precision']
Electric_bike_recall = evaluate_result['Electric bike']['recall']
plt.plot(Electric_bike_recall, Electric_bike_precision, label='Electric bike')



plt.xlabel("recall")
plt.ylabel("precision")
plt.title("Bike:%.3f  Electric bike:%.3f  " % (Bike_ap, Electric_bike_ap,))

ax.legend()
plt.show()







