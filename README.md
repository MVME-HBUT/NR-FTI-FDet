## **Introduction**

NR FTI-FDet is a real-time, NMS-free, and high accuracy detector in the RCBS-FD of freight trains.

Source code for **A Lightweight NMS-Free Framework for Real-Time Visual Fault Detection System of Freight Trains**. For more details, please refer to our [paper](https://ieeexplore.ieee.org/document/9779731).

The source code is based on [OneNet](https://github.com/PeizeSun/OneNet) and [Detectron2](https://github.com/facebookresearch/detectron2).


## Installation

### Requirements

- Linux or macOS with Python ≥ 3.6
- PyTorch ≥ 1.7 and [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation.
  Install them together at [pytorch.org](https://pytorch.org) to make sure of this
- OpenCV is optional but needed by demo and visualization

First install Detectron2 following the official guide: [INSTALL.md](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md).

*Please use Detectron2 with commit id [9eb4831](https://github.com/facebookresearch/detectron2/commit/9eb4831f742ae6a13b8edb61d07b619392fb6543) if you have any issues related to Detectron2.*

Then build LosNet with:

```
cd NR FTI-FDet
python setup.py build develop
```


Some projects may require special setup, please follow their own `README.md` in [configs](configs).

### Train Your Own Models

To train a model with "train_net.py", first
setup the corresponding datasets following
[datasets/README.md](https://github.com/facebookresearch/detectron2/blob/master/datasets/README.md),
then run:

```
pythonpath + ' ' + trainpy_path + ' ' + \
                        '--config-file' + ' ' + yamlpath + ' ' + \
                         'OUTPUT_DIR' + ' ' + outputpath
```

To evaluate the model after training, run:

```
pythonpath + ' ' + trainpy_path + ' ' + \
                        '--config-file' + ' ' + yamlpath + ' ' + \
                        '--eval-only' + ' '+ \
                        'MODEL.WEIGHTS' + ' ' + modelpath + ' ' + \
                        'OUTPUT_DIR' + ' ' + outputpath
```


Note that:

- The configs are made for 1-GPU training.
- ``trainpy_path`` is ``projects\project\train_net.py``


## **Cite**

```
@ARTICLE{9779731,  
author={Sun, Guodong and Zhou, Yang and Pan, Huilin and Wu, Bo and Hu, Ye and Zhang, Yang},  
journal={IEEE Transactions on Instrumentation and Measurement},   
title={{A Lightweight NMS-Free Framework for Real-Time Visual Fault Detection System of Freight Trains}},   
year={2022},  
volume={71},  
number={},  
pages={1-11},  
doi={10.1109/TIM.2022.3176901}}
```
