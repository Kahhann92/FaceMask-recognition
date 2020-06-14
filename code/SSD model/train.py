# -*- coding:utf-8 -*-
import pandas as pd
import os
import matplotlib.pyplot as plt
import csv
import numpy as np
import sys
import yaml
import math
import argparse
from keras.utils import plot_model
from keras.optimizers import Adam, SGD
# import keras.backend as K

from model_builder.backbone_model import ConvNet
from model_builder.ssd_builder import build_ssd_model
from loss.ssd_loss import loc_loss, cls_loss

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from utils.utils import PrintLearningRate
from utils.utils import generate_anchor_config

from data_process.pascal_loader import load_pascal_data
from data_process.data_generator import data_generator_multiprocess

from keras.utils import plot_model


Lambda = np.arange(1,401,1)
parser = argparse.ArgumentParser()
parser.add_argument("--config_path", help="Path to your config path.", type=str)
args = parser.parse_args()

config_path = args.config_path
config_path = './config/maskface.yaml'

if not os.path.exists(config_path):
    raise ValueError("Your config path is not exist.")

with open(config_path) as f:
    config = yaml.load(f)

gpu_id = config['gpu_id']
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
# 构建模型
detection_layer_config = config['detection_layer_config']
input_shape = config['input_shape']
num_class = config['num_class']
ssd_model = build_ssd_model(ConvNet, input_shape, detection_layer_config, num_class)

print(ssd_model.summary())

if config['optimizer']['name'] == "Adam":
    print("Adam Optimizer is selected")
    lr = config['optimizer'].get('lr', 0.001)
    decay_factor = config['optimizer'].get('decay_factor', 0.0)
    opt = Adam(lr=lr, decay=decay_factor)

else:
    print("SGD Optimizer is selected")
    opt = SGD(learning_rate=0.01, momentum=0.9)


model_save_path = config['model_save_path']
modelCkt = ModelCheckpoint(model_save_path, monitor="val_loss", verbose=1, save_best_only=True)
printLR = PrintLearningRate()

reduceLR = ReduceLROnPlateau(monitor="val_loss", factor=0.9, patience=3, verbose=1, min_lr=0.0000001)

ssd_model.compile(optimizer=opt,
                 loss=[loc_loss, cls_loss],
                 loss_weights=[1,1]
                 )


modelJson = ssd_model.to_json()
with open("models/model.json", 'w') as f:
    f.write(modelJson)


if config['fine_tune_path'] != 'None':
    print ("Start to restore weights from model.")
    ssd_model.load_weights(config['fine_tune_path'])
# plot_model(ssd_model, to_file="model_structure.png", show_shapes=True)  # 绘制模型

# 加载数据
trainset_path = config['trainset_path']
valset_path = config['valset_path']
class2id = config['class2id']

img_train, bboxes_train, class_train, difficult_train = load_pascal_data(trainset_path, class2id)
img_val, bboxes_val, class_val, difficult_val = load_pascal_data(valset_path, class2id)

anchor_config = generate_anchor_config(ssd_model, detection_layer_config)

batch_size = config['batch_size']

train_dg = data_generator_multiprocess(img_train,
                                       bboxes_train,
                                       class_train,
                                       difficult_train,
                                       train_mode=True,
                                       anchor_config = anchor_config,
                                       preprocess_config = config['preprocess'],
                                       input_shape = input_shape,
                                       num_class = num_class,
                                       batch_size = batch_size ,
                                       pool_nb = batch_size)

val_dg = data_generator_multiprocess(img_val,
                                       bboxes_val,
                                       class_val,
                                       difficult_val,
                                       train_mode=False,
                                       anchor_config = anchor_config,
                                       preprocess_config = config['preprocess'],
                                       input_shape = input_shape,
                                       num_class = num_class,
                                       batch_size = batch_size ,
                                       pool_nb = batch_size)


num_train_samples = len(img_train)
num_val_samples = len(img_val)

wwww=ssd_model.fit_generator(train_dg,
                        steps_per_epoch=math.ceil(num_train_samples / batch_size),
                        validation_data=val_dg,
                        validation_steps=math.ceil(num_val_samples / batch_size),
                        epochs=400,
                        callbacks=[modelCkt, reduceLR, printLR],
                        use_multiprocessing=False,
                        workers=1,
                        initial_epoch=config['fine_tune_epoch'])

# plot_model(ssd_model, to_file='./modelgraph.pdf')

print(wwww.history['loss'])
print(wwww.history['val_loss'])

theloss = pd.DataFrame(data=wwww.history['loss'])
# theloss.to_csv('./loss.csv', index=False, header=False)

thevalloss = pd.DataFrame(data=wwww.history['val_loss'])
# thevalloss.to_csv('./thevalloss.csv', index=False, header=False)

json_string = ssd_model.to_json()		# 方式1
open('./models/testmodel.json', 'w').write(json_string)
ssd_model.save_weights('./models/testmodel.h5')

# plt.figure()
# plt.plot(Lambda,theloss,label='train_loss')
# plt.plot(Lambda,thevalloss,label='Validation_mid')
# plt.legend()
# plt.savefig('./loss.png')
# plt.show()