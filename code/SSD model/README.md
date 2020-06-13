# Mask-Detection-Homework

## SSD model 压缩文件
其中为SSD模型的训练与测试所有代码，不包含训练数据。

## 环境配置
python 3.6  
pip install tensorflow-cpu==1.15    -i https://pypi.tuna.tsinghua.edu.cn/simple  
pip install keras    -i https://pypi.tuna.tsinghua.edu.cn/simple  
pip install python-opencv    -i https://pypi.tuna.tsinghua.edu.cn/simple  
 
## 数据下载与整理
AIZOO数据集 https://github.com/AIZOOTech/FaceMaskDetection  
其中包含train训练数据和val测试数据
在`config/maskface.yaml`文件中，修改  
trainset_path:  
  '/Dataset/FaceMaskNew/train' [AIZOO数据集中train训练数据的路径]   
valset_path:  
  '/Dataset/FaceMaskNew/val'   [AIZOO数据集中val训练数据的路径]  
为你的训练集和测试集路径  

## 训练模型
cd 拖入SSD model文件夹  
运行  
python train.py  
模型结构和训练好的模型参数存储在SSD model/models 中  
已经将训练好的模型放入，只需测试即可得到测试结果。

## 测试图片
将测试test-images/中的10张图片  
运行  
python test.py  
得到测试结果将在原图中显示是否存在口罩。 
