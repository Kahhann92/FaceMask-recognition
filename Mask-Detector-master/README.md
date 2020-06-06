# Mask-Detector

This repo contains a training setup for creating a model that can detect faces and label them as wearing\not wearing masks.

---

## Settings
* Framework: PyTorch
* Dataset: Custom build, by generating images with mask wearing faces from empty faces. (Used a modified script from https://github.com/prajnasb/observations)
 * [Faster RCNN](https://arxiv.org/abs/1506.01497) Architecture with pretrained MobileNet v2 backbone
 * To train: `python train_run_builder.py` (to change the training setup edit `config.py`)
 * To run the model on an image or on video stream from a local camera: `python detect.py --realtime=False\True`

### COCO Metrics

IoU metric: bbox
 | Score             |        IOU            |AREA         | RESULTS                |
 |:-----------------:|:---------------------:|:-----------:|:----------------------:|
 | Average Precision | (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.640  |
 | Average Precision | (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.990  |
 | Average Precision | (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.799  |
 | Average Precision | (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000 |
 | Average Precision | (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.609  |
 | Average Precision | (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.653  |
 | Average Recall    | (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.693  |
 | Average Recall    | (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.693  |
 | Average Recall    | (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.693  |
 | Average Recall    | (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000 |
 | Average Recall    | (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.650  |
 | Average Recall    | (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.706  |