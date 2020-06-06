## 框架逻辑
在配置文件（选用yaml配置文件格式）中配置模型定位层接出的节点名字，例如:
```
detection_layer_plugin_layers:
  conv5_3_relu:
    scale:
      - 0.1
      - 0.25
    aspect_ratio:
      - 1
      - 0.5
      - 2
```
anchor采用SSD论文中描述的方法，第一个尺度与所有的`aspect_ratio`配对，后面的尺度只与第一个`aspect_ratio`配对，所以对于每个
定位层，一共有`num(aspect_ratio) + num(scale) -1`组anchor

其他完整的模型训练和评估都在配置文件中配置。
## 一个额外的配置文件路径项
在 data_process/data_generator.py中，有个`attach_dir = 'D:/Projects/FaceMaskDetectionTrainWindows/attach_data/mask'`
要把这个路径设置为本文件夹的`attach_data/mask`所在路径，也就是前面要补上你这个工程的路径

在`config/maskface.yaml`文件中，修改
```
# Dataset part
trainset_path:
  'D:/Dataset/FaceMaskNew/train' 
valset_path:
  'D:/Dataset/FaceMaskNew/val'
```
为你的训练集和测试集路径
## 使用方法
1. 配置config.yaml文件
2. 运行 `python train.py --config-path /path/to/config.yaml`
3. 评估性能 `python evaluate_model.py --config_path /path/to/config.yaml --model_path path/to/your/model`
4. 推理测试 `python infer.py --config_path /path/to/config.yaml --img_path /path/to/your/img`
