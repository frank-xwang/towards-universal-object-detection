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

### prerequisites

* Python 2.7 or 3.6
* Pytorch 0.4.0
* CUDA 8.0 or higher

### Pretrained Model

You can download pre-trained models in ImageNet from:

* DA-50-5: [Dropbox](https://www.dropbox.com/s/iev3tkbz5wyyuz9/resnet101_caffe.pth?dl=0)
* DA-50-6: [Dropbox](https://www.dropbox.com/s/iev3tkbz5wyyuz9/resnet101_caffe.pth?dl=0)
* DA-50-8: [Dropbox](https://www.dropbox.com/s/iev3tkbz5wyyuz9/resnet101_caffe.pth?dl=0)
* DA-50-11: [Dropbox](https://www.dropbox.com/s/iev3tkbz5wyyuz9/resnet101_caffe.pth?dl=0)
* ResNet50: [Dropbox](https://www.dropbox.com/s/iev3tkbz5wyyuz9/resnet101_caffe.pth?dl=0)

Download them and put them into the data/pretrained_model/.

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
export C_INCLUDE_PATH=/usr/local/cuda-9.0/include:${C_INCLUDE_PATH}
export PATH=/usr/local/cuda-9.0/bin${PATH:+:${PATH}}
export LIBRARY_PATH=/usr/local/cuda-9.0/lib64:$LIBRARY_PATH
conda install pytorch=0.4.0 cuda90 cudatoolkit==9.0 -c pytorch
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
  | TitanX (Maxwell/Pascal) | sm_52 |
  | GTX 960M | sm_50 |
  | GTX 1080 (Ti) | sm_61 |
  | Grid K520 (AWS g2.2xlarge) | sm_30 |
  | Tesla K80 (AWS p2.xlarge) | sm_37 |

More details about setting the architecture can be found [here](https://developer.nvidia.com/cuda-gpus) or [here](http://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/)

## Train

Before training, set the right directory to save and load the trained models. Change the arguments "save_dir" and "load_dir" in trainval_net.py and test_net.py to adapt to your environment.

To train a faster R-CNN model with vgg16 on pascal_voc, simply run:
```
CUDA_VISIBLE_DEVICES=$GPU_ID python trainval_net.py \
                   --dataset pascal_voc --net da-50 \
                   --bs $BATCH_SIZE --nw $WORKER_NUMBER \
                   --lr $LEARNING_RATE --lr_decay_step $DECAY_STEP \
                   --cuda
```
where 'bs' is the batch size with default 1. Alternatively, to train with resnet101 on pascal_voc, simple run:
```
 CUDA_VISIBLE_DEVICES=$GPU_ID python trainval_net.py \
                    --dataset pascal_voc --net da-50 \
                    --bs $BATCH_SIZE --nw $WORKER_NUMBER \
                    --lr $LEARNING_RATE --lr_decay_step $DECAY_STEP \
                    --cuda
```
Above, BATCH_SIZE and WORKER_NUMBER can be set adaptively according to your GPU memory size. **On Titan Xp with 12G memory, it can be up to 4**.

If you have multiple (say 8) Titan Xp GPUs, then just use them all! Try:
```
python trainval_net.py --dataset pascal_voc --net da-50 \
                       --bs 24 --nw 8 \
                       --lr $LEARNING_RATE --lr_decay_step $DECAY_STEP \
                       --cuda --mGPUs

```

Change dataset to "coco" or 'vg' if you want to train on COCO or Visual Genome.

## Test

If you want to evaluate the detection performance of each datasets, download pre-trained model and put it in models/da_res50/universal/, then simply run:
```
bash scripts/test_universal.sh
```
Specify the specific GPU device ID(GPU_ID), network(net), data directory(DATA_DIR), number of adapters(num_adapters), model session(SESSION), checkepoch(EPOCH), checkpoint iterations(CHECKPOINT), datasets to test(datasets) and etc. before running test_universal.sh file. Only sigle GPU testing is supported.

Pre-trained model will be named as faster_rcnn_universal_SESSION_EPOCH_CHECKPOINT.pth

Results and models:

  | #Adapter | less_blocks | KITTI | VOC | Widerface | LISA | Kitchen | COCO | DOTA | DeepLesion | Comic | Clipart | Watercolor | AVG | Model |
  | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
  | 11 | False  | 68.1 | 82.0 | 51.6 | 88.3 | 90.1 | 46.5 | 57.0 | 57.3 | 50.7 | 53.1 | 58.4 | 63.8 | [model](https://drive.google.com/file/d/1zh-WguXt3b1IKjfeyFWIOqOxLFflbghL/view?usp=sharing) |

### Some popular problems
1. fatal error: cuda.h: No such file or directory:

    Export C_INCLUDE_PATH=/usr/local/cuda-9.0/include:${C_INCLUDE_PATH}, then run "sh make.sh"

2. RuntimeError: CUDNN_STATUS_EXECUTION_FAILED:
    
    Usually, this is caused by using different cudnn when building and running pytorch. You can check this simply by running: torch.backends.cudnn.version(). You can also test by checking if the output of "pytorch.version.cuda" and "nvcc --version" gives you the same cudnn version. If the above checks fail, you need to reinstall pytorch and make sure to use the same cudnn within the inference time.
    
3. THCudaCheck FAIL file=/opt/conda/conda-bld/pytorch_1524586445097/work/aten/src/THC/THCGeneral.cpp line=844 error=11 : invalid argument

    This error will appear for RTX2080 GPU cards with cuda9.x. This error will not have influence during inference and training time for pytorch0.4.0, we can simly ignore it. Check [issue](https://github.com/pytorch/pytorch/issues/21154) and [issue](https://discuss.pytorch.org/t/thcudacheck-fail-file-pytorch-aten-src-thc-thcgeneral-cpp/31788/13) for details.

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
