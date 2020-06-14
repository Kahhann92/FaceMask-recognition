# 文件夹内容简介
此一project是由清华大学医学院的林非凡与郑家瀚共同开发完成，这里运用了三个目标检测模型，来找到图像里的人脸，以及他们是否有带口罩，这一readme.md文件是为了帮助使用者如何正确使用我们的code。

## SSD model 
其中为SSD模型的训练与测试所需的所有代码，这里不包含训练数据。

## Faster-RCNN
其中为Faster-RCNN模型的训练与测试的ipynb文件，也有摄像头实时目标检测的文件，不包含训练数据。

## Focal-Loss
其中为Focal-Loss模型的训练与测试的ipynb文件，不包含训练数据。

## test-images
这里提供十张照片供各模型测试。

# 环境配置
## SSD model 
python 3.6  
pip install tensorflow-cpu==1.15    -i https://pypi.tuna.tsinghua.edu.cn/simple  
pip install keras    -i https://pypi.tuna.tsinghua.edu.cn/simple  
pip install python-opencv    -i https://pypi.tuna.tsinghua.edu.cn/simple  
pip install -r requirements.txt

## Faster-RCNN & Focal-Loss
这里主要使用的是ipynb文件，是基于google colab线上jupyter notebook开发出来的。主要是用方法有二，一是直接连接到google colab的文件去运行，二是在裸机搭配出与google colab相匹配的环境去运行。若是使用方法一的话，裸机无需装配任何函数库，也无需什么GPU，因此这里优先推荐方法一。

## google colab运行
google colab的环境搭配都在ipynb的前面部分完成，只要逐步运行就行，若要使用google colab运行，这里不必再配置环境。

## 裸机运行

### 先装 dependencies

Python 3.6.9
pip install -U torch==1.5 torchvision==0.6 -f https://download.pytorch.org/whl/cu101/torch_stable.html 
pip install cython pyyaml==5.1
pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'


### Build Detectron2 from Source
这里的讯息采自https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md，欲知更多详情，请参阅这一网站。

gcc & g++ ≥ 5 . [ninja](https://ninja-build.org/) is recommended for faster build.
After having them, run:
```
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
# (add --user if you don't have permission)

# Or, to install it from a local clone:
git clone https://github.com/facebookresearch/detectron2.git
python -m pip install -e detectron2

# Or if you are on macOS
# CC=clang CXX=clang++ python -m pip install -e .
```

To __rebuild__ detectron2 that's built from a local clone, use `rm -rf build/ **/*.so` to clean the
old build first. You often need to rebuild detectron2 after reinstalling PyTorch.

### Install Pre-Built Detectron2 (Linux only)

Choose from this table:

<table class="docutils"><tbody><th width="80"> CUDA </th><th valign="bottom" align="left" width="100">torch 1.5</th><th valign="bottom" align="left" width="100">torch 1.4</th> <tr><td align="left">10.2</td><td align="left"><details><summary> install </summary><pre><code>python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu102/torch1.5/index.html
</code></pre> </details> </td> <td align="left"> </td> </tr> <tr><td align="left">10.1</td><td align="left"><details><summary> install </summary><pre><code>python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.5/index.html
</code></pre> </details> </td> <td align="left"><details><summary> install </summary><pre><code>python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.4/index.html
</code></pre> </details> </td> </tr> <tr><td align="left">10.0</td><td align="left"> </td> <td align="left"><details><summary> install </summary><pre><code>python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu100/torch1.4/index.html
</code></pre> </details> </td> </tr> <tr><td align="left">9.2</td><td align="left"><details><summary> install </summary><pre><code>python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu92/torch1.5/index.html
</code></pre> </details> </td> <td align="left"><details><summary> install </summary><pre><code>python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu92/torch1.4/index.html
</code></pre> </details> </td> </tr> <tr><td align="left">cpu</td><td align="left"><details><summary> install </summary><pre><code>python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch1.5/index.html
</code></pre> </details> </td> <td align="left"><details><summary> install </summary><pre><code>python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch1.4/index.html
</code></pre> </details> </td> </tr></tbody></table>


Note that:
1. The pre-built package has to be used with corresponding version of CUDA and official PyTorch release.
   It will not work with a different version of PyTorch or a non-official build of PyTorch.
2. Such installation is out-of-date w.r.t. master branch of detectron2. It may not be
   compatible with the master branch of a research project that uses detectron2 (e.g. those in
   [projects](projects) or [meshrcnn](https://github.com/facebookresearch/meshrcnn/)).

### for macOS

### Linux only
Pillow== 7.0.0 
numpy                    1.18.5


 
# 数据下载与整理
AIZOO数据集 https://github.com/AIZOOTech/FaceMaskDetection  
其中包含train训练数据和val测试数据
在`config/maskface.yaml`文件中，修改  
trainset_path:  
  '/Dataset/FaceMaskNew/train' [AIZOO数据集中train训练数据的路径]   
valset_path:  
  '/Dataset/FaceMaskNew/val'   [AIZOO数据集中val训练数据的路径]  
为你的训练集和测试集路径  

# 训练模型

cd 拖入SSD model文件夹  
运行  python train.py  
模型结构和训练好的模型参数存储在SSD model/models 中  
已经将训练好的模型放入，只需测试即可得到测试结果。

# 测试图片
将测试test-images/中的10张图片  
运行  
python test.py  
得到测试结果将在原图中显示是否存在口罩。 
