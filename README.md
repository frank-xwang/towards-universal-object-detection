# Towards Universal Object Detection by Domain Attention.

by Xudong Wang, Zhaowei Cai, Dashan Gao and Nuno Vasconcelos in UC San Diego and 12 Sigma Technologies.

This project is based on Pytorch reproduced Faster R-CNN by [jwyang](https://github.com/jwyang/faster-rcnn.pytorch)

### Project Pages
http://www.svcl.ucsd.edu/projects/universal-detection/

### Introduction
This is the benchmark introduced in CVPR 2019 paper: [Towards Universal Object Detection by Domain Attention](https://arxiv.org/pdf/1904.04402.pdf). The goal of this benchmark is to encourage designing universal object detection system, capble of solving various detection tasks. To train and evaluate universal/multi-domain object detection systems, we established a new universal object detection benchmark (UODB) of 11 datasets. Details of UODB can be obtained from project pages. You can also download these datasets in that project page.

### Datasets Preparation

First of all, clone the code

Then, create a folder:
```
cd towards-universal-object-detection && mkdir data
```

Then put all the donwloaded datasets from [UODB benchmark](http://www.svcl.ucsd.edu/projects/universal-detection/) inside data folder and unzip all of them.

All VOC format datasets should be in the structure of:

    datasets
        --Annotations
            --0.xml
        --ImageSets
            --Main
        --JPEGImages
            --0.jpg or 0.png

### prerequisites

* Python 3.6
* Pytorch 0.4.0
* CUDA 8.0 or higher

### Pretrained Model

You can download pre-trained models in ImageNet from:

* DA-50: [Dropbox](https://drive.google.com/file/d/1kddC55_eByFfMZqDTM9cLj0j1BiHBq9D/view?usp=sharing)
* ResNet50: [Dropbox](https://drive.google.com/file/d/1_0wFe2soxLkyP5DCCpOJddp1k_xcowv-/view?usp=sharing)

Download and unzip DAResNet50.zip, and put them into data/pretrained_model/ folder.

### Compilation

1. Create virtual envrionment and activate it:

```
conda create -n uodb python=3.6 -y
conda activate uodb
```

2. Install all the python dependencies using pip:
```
pip install -r requirements.txt --user
```

3. Install pytorch0.4.0 with conda:
```
conda install pytorch=0.4.0 cuda80 cudatoolkit==8.0 -c pytorch
```
Please change cuda version accordingly.

4. Compile the cuda dependencies using following simple commands:

```
cd lib
sh make.sh
```
As pointed out by [ruotianluo/pytorch-faster-rcnn](https://github.com/ruotianluo/pytorch-faster-rcnn), choose the right `-arch` in `make.sh` file, to compile the cuda code:

  | GPU model  | Architecture |
  | ------------- | ------------- |
  | Tesla K80 (AWS p2.xlarge) | sm_37 |
  | TitanX (Maxwell/Pascal) | sm_52 |
  | GTX 960M | sm_50 |
  | GTX 1080 (Ti) | sm_61 |
  | Grid K520 (AWS g2.2xlarge) | sm_30 |
  | RTX 2080 (Ti) | sm_70 |

More details about setting the architecture can be found [here](https://developer.nvidia.com/cuda-gpus) or [here](http://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/)

## Train

Before training, set the right directory to save and load the trained models. Change the arguments "save_dir" and "load_dir" in universal_model.py to adapt to your environment.

To train a model with 11 adapters, simply run:
```
bash scripts/train_universal.sh
```
Above, BATCH_SIZE and WORKER_NUMBER can be set adaptively according to your GPU memory size. Specify the specific GPU device ID(GPU_ID), network(net), data directory(DATA_DIR), number of adapters(num_adapters), model session(SESSION), checkepoch(EPOCH), checkpoint iterations(CHECKPOINT), list of datasets to train(datasets_list), using less domain attention blocks(less_blocks) and etc. before running train_universal.sh file.

## Test

If you want to evaluate the detection performance of each datasets, download pre-trained model and put it in models/da_res50/universal/, then simply run:
```
bash scripts/test_universal.sh
```
Specify the specific GPU device ID(GPU_ID), network(net), data directory(DATA_DIR), number of adapters(num_adapters), model session(SESSION), checkepoch(EPOCH), checkpoint iterations(CHECKPOINT), datasets to test(datasets) and etc. before running test_universal.sh file. Only sigle GPU testing is supported.

Pre-trained model will be named as faster_rcnn_universal_SESSION_EPOCH_CHECKPOINT.pth

Results and models for 5 datasets universal model:

  | #Adapter | less_blocks | KITTI | VOC | Widerface | LISA | Kitchen | AVG | download |
  | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
  | 3 | True  | 68.0 | 78.8 | 51.9 | 88.1 | 87.1 | 74.8 | [model](https://drive.google.com/file/d/1CbbdBHmyIoALOTBSjZzEKPDHK9QKdUus/view?usp=sharing) |
  | 5 | True  | 67.9 | 79.2 | 52.2 | 87.5 | 88.5 | 75.1 | [model](https://drive.google.com/file/d/1x5Rd33yeUXicOEXH6TIqhfYBX8wqmDRI/view?usp=sharing) |
  | 7 | True  | 68.2 | 79.9 | 52.1 | 89.7 | 88.0 | 75.6 | [model](https://drive.google.com/file/d/1gLQZGn6Vb-AzfzFmf7YCZLqE_Mfwzyq6/view?usp=sharing) |

Results and models for 11 datasets universal model:

  | #Adapter | less_blocks | KITTI | VOC | Widerface | LISA | Kitchen | COCO | DOTA | DeepLesion | Comic | Clipart | Watercolor | AVG | download |
  | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
  | 6 | True  | 67.6 | 82.7 | 51.8 | 87.9 | 88.7 | 46.8 | 57.0 | 54.8 | 52.6 | 54.6 | 58.2 | 63.9 | [model](https://drive.google.com/file/d/1uyQ-BX_p8T3HEaJrsvUTshmnIy80Et5M/view?usp=sharing) |
  | 8 | True  | 68.0 | 82.4 | 51.3 | 87.6 | 90.0 | 47.0 | 56.3 | 53.4 | 53.4 | 55.8 | 60.6 | 64.2 | [model](https://drive.google.com/file/d/1WtthQFm_SEbMVcQnZnD8Xgm3n-msDw1B/view?usp=sharing) |

### Some popular problems
1. fatal error: cuda.h: No such file or directory:

    Export C_INCLUDE_PATH=/usr/local/cuda-8.0/include:${C_INCLUDE_PATH}, then run "sh make.sh"

2. RuntimeError: CUDNN_STATUS_EXECUTION_FAILED:
    
    Usually, this is caused by using different cudnn when building and running pytorch. You can check this simply by running: torch.backends.cudnn.version(). You can also test by checking if the output of "pytorch.version.cuda" and "nvcc --version" gives you the same cudnn version. If the above checks fail, you need to reinstall pytorch and make sure to use the same cudnn within the inference time.
    
3. THCudaCheck FAIL file=/opt/conda/conda-bld/pytorch_1524586445097/work/aten/src/THC/THCGeneral.cpp line=844 error=11 : invalid argument

    This error will appear for RTX2080 GPU cards with cuda8.x or cuda9.x, you may need to install pytorch from source to solve it. Check [issue](https://github.com/pytorch/pytorch/issues/15797) and [issue](https://discuss.pytorch.org/t/thcudacheck-fail-file-pytorch-aten-src-thc-thcgeneral-cpp/31788/13) for details. This error can be ignored within inference time.

If you meet any problems, please feel free to contact me by: frank.xudongwang@gmail.com

### Citation

If you use our code/model/data, please cite our paper:

    @inproceedings{wang2019towards,
      title={Towards universal object detection by domain attention},
      author={Wang, Xudong and Cai, Zhaowei and Gao, Dashan and Vasconcelos, Nuno},
      booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
      pages={7289--7298},
      year={2019}
    }
