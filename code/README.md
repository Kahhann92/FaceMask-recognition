
# 文件夹内容简介

此一project是由清华大学医学院的林非凡与郑家瀚共同开发完成，这里运用了三个目标检测模型，来找到图像里的人脸，以及他们是否有带口罩，这一readme.md文件是为了帮助使用者如何正确使用我们的code。

下面介绍一下主要的几个子文件夹。  

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

此一模型原在无GPU的mac笔电、python 3.6的环境运行，请先在terminal运行以下code。

```
pip install tensorflow-cpu==1.15  -i https://pypi.tuna.tsinghua.edu.cn/simple

pip install keras  -i https://pypi.tuna.tsinghua.edu.cn/simple

pip install python-opencv  -i https://pypi.tuna.tsinghua.edu.cn/simple

pip install -r requirements.txt
```
  

## Faster-RCNN & Focal-Loss

这里主要使用的是ipynb文件，是基于google colab线上jupyter notebook开发出来的。主要是用方法有二，一是直接连接到google colab的文件去运行，二是在裸机搭配出与google colab相匹配的环境去运行。若是使用方法一的话，裸机无需装配任何函数库，也无需什么GPU，因此这里优先推荐方法一。

  

### google colab运行

google colab的环境搭配都在ipynb的前面部分完成，只要逐步运行就行，若要使用google colab运行，这里不必再配置环境，只需要点击下面会提供的链接，就可以开启了。

若只是要检测成果，打开链接以后，可以点击主项目栏上的"File"，然后点击"Open in Playground"就可以正常运行了。

若要做编辑，则需要File > Save a copy in GitHub或者  Save a copy in drive。后者会需要用到google云盘。

### 裸机运行

#### 先装 dependencies

此一模型原在Google-colab、Python 3.6.9的环境运行，请先在terminal运行以下code。
```
pip install -U torch==1.5 torchvision==0.6 -f https://download.pytorch.org/whl/cu101/torch_stable.html

pip install cython pyyaml==5.1

pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'

pip install numpy==1.18.5

pip install google-colab

pip install jsonschema==2.6.0

pip install tensorflow==2.2.0

pip install zipfile36
```
#### Build Detectron2 from Source
这里的讯息采自 [Detectron2](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md)，欲知更多详情，请参阅这一网站。
gcc & g++ ≥ 5 are required. [ninja](https://ninja-build.org/) is recommended for faster build.

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

  

To __**rebuild**__ detectron2 that's built from a local clone, use `rm -rf build/ **/*.so` to clean the old build first. You often need to rebuild detectron2 after reinstalling PyTorch.

  

#### Install Pre-Built Detectron2 (Linux only)

  

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

1. The pre-built package has to be used with corresponding version of CUDA and official PyTorch release. It will not work with a different version of PyTorch or a non-official build of PyTorch.

2. Such installation is out-of-date w.r.t. master branch of detectron2. It may not be compatible with the master branch of a research project that uses detectron2 (e.g. those in [projects](projects) or [meshrcnn](https://github.com/facebookresearch/meshrcnn/)).

  

# 数据下载与整理

这部分将会引导您如何下载、整理数据，以让各模型能够顺利进行训练，来还原我们的成果。

## 数据准备（给训练用）
### SSD
[AIZOO数据集](https://github.com/AIZOOTech/FaceMaskDetection) 其中包含train训练数据和val测试数据在`config/maskface.yaml`文件中，修改

trainset_path:
'/Dataset/FaceMaskNew/train' [AIZOO数据集中train训练数据的路径]

valset_path:
'/Dataset/FaceMaskNew/val' [AIZOO数据集中val训练数据的路径]

为你的训练集和测试集路径

### Faster RCNN & Focal Loss

无需另外下载，只要成功运行code，即会自动下载

## 数据格式转换

这里介绍我们如何作数据格式转换。

### VOC2COCO

这里会运行的code是把原数据集voc的格式整理、转换成可供Faster-RCNN以及Focal Los使用的COCO格式。如您已经用terminal cd到已经解压的文件里头之后，请cd到test-images里。
```
cd code/test-images
```
然后再运行这一python 文件。
```
python voc2coco.py
```


# 训练模型

这部分将会引导您如何训练各种模型，各模型的epoch数都已经调到合适的数量，适合作快速检测。

## SSD
 1. cd 拖入SSD model文件夹
 2. 运行  python train.py
 3. 模型结构和训练好的模型参数存储在SSD model/models 中已经将训练好的模型放入，只需测试即可得到测试结果。

## Faster RCNN

可使用google colab运行faster_RCNN.ipynb，请点击此[链接](https://colab.research.google.com/drive/1ZNtf167xZgS6wL3XJV1qs5jOYnpSzwDA?usp=sharing)，会连接到我们的google云盘的google colab文件，即可逐步运行。(推荐)

or 

使用 jupyter notebook开启Faster-RCNN/faster_RCNN.ipynb，逐步运行。

## Focal Loss

可使用google colab运行Focal_Loss.ipynb，请点击此[链接](https://colab.research.google.com/drive/10NaDQzf2QlF_GlqKIvYtoNjVJZhslWmB?usp=sharing)，会连接到我们的google云盘的google colab文件，即可逐步运行。(推荐)

or 

使用 jupyter notebook开启Focal-Loss/Focal_Loss.ipynb，逐步运行。
  

# 测试模型

这部分将会引导您如何使用各种模型作测试，这里会有三个模型个别作图形检测，以及一个用Faster RCNN来做摄像头实时检测的示范。

## 图片测试

这里会让模型对test-images里的十张图片进行检测，这里的十张照片有五张是口罩，有五张是没口罩的，包含各种难易度，运行完毕，你就可以大概看出一个高下。

### SSD

将测试test-images/中的10张图片，请运行
```
python test.py
```
得到测试结果将在原图中显示是否存在口罩。

### Faster RCNN

可使用google colab运行Test_10_images_with_Faster_RCNN.ipynb，请点击此[链接](https://colab.research.google.com/drive/1rPW8QzUWKu1d9h9-kwhugdstJAovs0wZ?usp=sharing)，会连接到我们的google云盘的google colab文件，即可逐步运行。(推荐)

or 

使用 jupyter notebook开启 Faster-RCNN/Test_10_images_with_Faster_RCNN.ipynb，逐步运行。

### Focal Loss

可使用google colab运行Test_10_images_with_Focal_Loss.ipynb，请点击此[链接](https://colab.research.google.com/drive/1Uq8yVkv_0uzQmib6udEWcHZrLymZP_k-?usp=sharing)，会连接到我们的google云盘的google colab文件，即可逐步运行。(推荐)

or 

使用 jupyter notebook开启 Focal-Loss/Test_10_images_with_Focal_Loss.ipynb，逐步运行。


## 摄像头实时监测

可使用google colab运行webcam_with_Faster_RCNN.ipynb，请点击此[链接](https://colab.research.google.com/drive/1zNjPg_udwPuF-nr4-sEgAZLLjkVVcGeb?usp=sharing)，会连接到我们的google云盘的google colab文件，即可逐步运行。

