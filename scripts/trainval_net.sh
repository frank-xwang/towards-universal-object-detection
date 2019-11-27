BATCH_SIZE=8
net=se_res50
WORKER_NUMBER=1
LEARNING_RATE=0.01
DECAY_STEP=10
GPU_ID=3,5,6,7
checkepoch=10
SESSION=133
CHECKPOINT=1608

#number of epochs to train
epochs=14
resume=True
#datasets=coco
#datasets=comic
#datasets=watercolor
datasets=citypersons
datasets=caltech
datasets=KITTI
datasets=deeplesion
datasets=dota
datasets=coco
datasets=widerface
datasets=Kitchen
datasets=LISA
datasets=cross_domain
datasets=pascal_voc_0712
datasets=watercolor

fix_bn=True
USE_FLIPPED=True
DATA_DIR=data
warmup_steps=0

CUDA_VISIBLE_DEVICES=${GPU_ID} python trainval_net.py \
                    --dataset ${datasets} --net ${net} \
                    --bs ${BATCH_SIZE} --nw ${WORKER_NUMBER} \
                    --lr ${LEARNING_RATE} --lr_decay_step ${DECAY_STEP} \
                    --cuda --mGPUs \
                    --USE_FLIPPED ${USE_FLIPPED} \
                    --DATA_DIR ${DATA_DIR} \
                    --fix_bn ${fix_bn} \
                    --s ${SESSION} \
                    --warmup_steps ${warmup_steps} \
                    --epochs ${epochs}
                    --r ${resume} \
                    --checksession ${SESSION} \
                    --checkepoch ${checkepoch} \
                    --checkpoint ${CHECKPOINT}