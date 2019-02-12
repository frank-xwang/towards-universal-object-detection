BATCH_SIZE=16
net=se_res50
WORKER_NUMBER=1
LEARNING_RATE=0.01
DECAY_STEP=10
GPU_ID=0,1,2,3,4,5,6,7
checkepoch=1
SESSION=127
CHECKPOINT=2

#number of epochs to train
epochs=20
resume=True
datasets=KAISTVOC
#datasets=coco
datasets=clipart
#datasets=comic
#datasets=watercolor
datasets=citypersons
#datasets=dota
datasets=caltech
datasets=cross_domain
datasets=deeplesion
datasets=Kitchen
datasets=coco
datasets=pascal_voc
datasets=KITTIVOC
datasets=LISA
datasets=widerface

fix_bn=True
USE_FLIPPED=True
DATA_DIR=/home/Xwang/HeadNode-1/universal_model_/data

CUDA_VISIBLE_DEVICES=${GPU_ID} python trainval_net.py \
                    --dataset ${datasets} --net ${net} \
                    --bs ${BATCH_SIZE} --nw ${WORKER_NUMBER} \
                    --lr ${LEARNING_RATE} --lr_decay_step ${DECAY_STEP} \
                    --cuda --mGPUs \
                    --USE_FLIPPED ${USE_FLIPPED} \
                    --DATA_DIR ${DATA_DIR} \
                    --fix_bn ${fix_bn} \
                    --s ${SESSION} \
                    --epochs ${epochs}
                    --r ${resume} \
                    --checksession ${SESSION} \
                    --checkepoch ${checkepoch} \
                    --checkpoint ${CHECKPOINT}
