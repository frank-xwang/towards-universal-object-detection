BATCH_SIZE=16
net=vgg16_bn_finetune
net=res18_finetune
WORKER_NUMBER=9
LEARNING_RATE=0.01
DECAY_STEP=8
GPU_ID=4,5,6,8
GPU_ID=1,3
#GPU_ID=3,4
#GPU_ID=0,1,2,3
checkepoch=2
SESSION=35
CHECKPOINT=1608
#number of epochs to train
epochs=16
resume=True
backward_together=False
finetuneBN_DS=0
finetuneBN_linear=0
fa_conv_num=2

add_filter_num=20 # set 0 if do not add filters
add_filter_ratio=0.1
# Info of fine_tune vgg16_bn
# PRE_net=vgg16_bn
# PRE_datasets=pascal_voc_0712
# PRE_epoch=11
# PRE_ses=10
# PRE_checkpoint=8274

# Info of fine_tune res18

PRE_net=res18
PRE_datasets=pascal_voc_0712
PRE_epoch=14
PRE_ses=2
PRE_checkpoint=1378

#datasets=caltech
datasets=KITTIVOC
#datasets=pascal_voc_0712
#datasets=coco
#datasets=universal
#datasets=widerface
CUDA_VISIBLE_DEVICES=${GPU_ID} python fine_tune.py \
                    --dataset ${datasets} --net ${net} \
                    --bs ${BATCH_SIZE} --nw ${WORKER_NUMBER} \
                    --lr ${LEARNING_RATE} --lr_decay_step ${DECAY_STEP} \
                    --cuda --mGPUs \
                    --fa_conv_num ${fa_conv_num} \
                    --finetuneBN_DS ${finetuneBN_DS} \
                    --finetuneBN_linear ${finetuneBN_linear} \
                    --s ${SESSION} \
                    --pre_net ${PRE_net} \
                    --pre_datasets ${PRE_datasets} \
                    --pre_ses ${PRE_ses} \
                    --pre_epoch ${PRE_epoch} \
                    --pre_checkpoint ${PRE_checkpoint} \
                    --add_filter_num ${add_filter_num} \
                    --add_filter_ratio ${add_filter_ratio}\
                    --checksession ${SESSION} \
                    --checkepoch ${checkepoch} \
                    --checkpoint ${CHECKPOINT} \
                    --epochs ${epochs} 
                    --r ${resume} \
