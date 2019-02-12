BATCH_SIZE=2
net=vgg16_bn_finetune
net=res18_finetune
WORKER_NUMBER=8
LEARNING_RATE=0.001
DECAY_STEP=5
GPU_ID=1
EPOCH=15
SESSION=28
CHECKPOINT=466

finetuneBN_DS=0
finetuneBN_linear=0
fa_conv_num=2
add_filter_num=20 # set 0 if do not add filters
add_filter_ratio=0.3

PRE_datasets=pascal_voc_0712
PRE_epoch=11
PRE_ses=10
PRE_checkpoint=8274
datasets=pascal_voc_0712
PRE_net=vgg16_bn
#datasets=caltech
datasets=KITTIVOC
#datasets=coco
#datasets=universal
#datasets=widerface

# Info of fine_tune res18
PRE_net=res18
PRE_datasets=pascal_voc_0712
PRE_epoch=14
PRE_ses=2
PRE_checkpoint=1378

CUDA_VISIBLE_DEVICES=${GPU_ID} python test_fine_tune.py \
                    --dataset ${datasets} --net ${net} \
                    --checksession ${SESSION} \
                    --checkepoch ${EPOCH} \
                    --checkpoint ${CHECKPOINT} \
                    --cuda --mGPUs \
                    --finetuneBN_DS ${finetuneBN_DS} \
                    --finetuneBN_linear ${finetuneBN_linear} \
                    --fa_conv_num ${fa_conv_num} \
                    --add_filter_num ${add_filter_num} \
                    --add_filter_ratio ${add_filter_ratio} \
                    --pre_datasets ${PRE_datasets} \
                    --pre_ses ${PRE_ses} \
                    --pre_net ${PRE_net} \
                    --pre_epoch ${PRE_epoch} \
                    --pre_checkpoint ${PRE_checkpoint} \
