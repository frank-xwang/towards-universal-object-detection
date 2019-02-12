BATCH_SIZE=3
net=vgg16_bn
WORKER_NUMBER=6
LEARNING_RATE=0.004
DECAY_STEP=9
GPU_ID=6
checkepoch=8
SESSION=10
CHECKPOINT=1869

#number of epochs to train
epochs=16
resume=True
#datasets=pascal_voc_0712
datasets=KAISTVOC
#datasets=coco
#datasets=clipart
#datasets=comic
#datasets=watercolor
datasets=caltech
datasets=pascal_voc
datasets=pascal_voc_0712
datasets=widerface
datasets=KITTIVOC
datasets=citypersons

CUDA_VISIBLE_DEVICES=${GPU_ID} python compression_train.py \
                    --dataset ${datasets} --net ${net} \
                    --bs ${BATCH_SIZE} --nw ${WORKER_NUMBER} \
                    --lr ${LEARNING_RATE} --lr_decay_step ${DECAY_STEP} \
                    --cuda --mGPUs \
                    --s ${SESSION} \
                    --epochs ${epochs} 
                    --r ${resume} \
                    --checksession ${SESSION} \
                    --checkepoch ${checkepoch} \
                    --checkpoint ${CHECKPOINT}