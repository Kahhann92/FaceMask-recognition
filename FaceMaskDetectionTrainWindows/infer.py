# -*- coding:utf-8 -*-
import cv2
import os
import yaml
import argparse
from keras.models import load_model
from loss.ssd_loss import loc_loss, cls_loss
from inference.infer_utils import init_config, sample_inference

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", help="Path to your config path.", type=str)
parser.add_argument("--img_path", help="Path to the test image path", type=str)
parser.add_argument("--model_path", help="Path to the model path", type=str)
args = parser.parse_args()

config_path = args.config_path
img_path = args.img_path
model_path = args.model_path

# config_path = args.config_path
# config_path = 'config/example.yaml'

# model_path = "models/kedetion_model_depther_004_val_loss_3.8696_loc_loss_1.7035_cls_loss_1.9369.hdf5"
with open(config_path) as f:
    config = yaml.load(f)

gpu_id = config['gpu_id']

os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id

model = load_model(model_path, custom_objects={"loc_loss":loc_loss, "cls_loss":cls_loss})
init_config(model, config_path)
print(model.summary())
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
sample_inference(img, model, show_result=True,target_shape=(260,260))

