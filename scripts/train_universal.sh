<<<<<<< HEAD
GPU_ID=0,1,2,3,4,5,6,7
=======
>>>>>>> 793eeda709a4483589939795954491531204c768
BATCH_SIZE=16
net=data_att_res50
WORKER_NUMBER=4
LEARNING_RATE=0.01
DECAY_STEP=10
<<<<<<< HEAD
SAVE_SESSION=11083
epochs=20
backward_together=0 # 0: independent; 1: together
USE_FLIPPED=1 # choose 1 for using flipped images, 0 for don't
datasets=universal
DATA_DIR=data
random_resize=True
=======
GPU_ID=0,1,2,3
checkepoch=7
SAVE_SESSION=11083
checksession=11083
CHECKPOINT=5720
epochs=20
resume=True
backward_together=0 # 0: independent; 1: together
USE_FLIPPED=1 # choose 1 for using flipped images, 0 for don't
# /home/Xwang/HeadNode-1/universal_model_/models/data_att_res50/universal/faster_rcnn_universal_11086_9_5720.pth
datasets=universal
DATA_DIR=/home/Xwang/HeadNode-1/universal_model_/data

random_resize=False
>>>>>>> 793eeda709a4483589939795954491531204c768
fix_bn=True
use_bn_mux=False
update_chosen=False
randomly_chosen_datasets=True
warmup_steps=0
<<<<<<< HEAD
num_se=11
less_blocks=False
# resume=True
# checkepoch=7
# checksession=11083
# CHECKPOINT=5720
=======
rpn_univ=False
num_se=3
less_blocks=True
>>>>>>> 793eeda709a4483589939795954491531204c768

CUDA_VISIBLE_DEVICES=${GPU_ID} python universal_model.py \
                    --dataset ${datasets} --net ${net} \
                    --bs ${BATCH_SIZE} --nw ${WORKER_NUMBER} \
                    --lr ${LEARNING_RATE} --lr_decay_step ${DECAY_STEP} \
                    --USE_FLIPPED ${USE_FLIPPED} \
                    --cuda --mGPUs \
                    --s ${SAVE_SESSION} \
                    --epochs ${epochs} \
                    --DATA_DIR ${DATA_DIR} \
                    --random_resize ${random_resize} \
                    --num_se ${num_se} \
                    --less_blocks ${less_blocks} \
                    --fix_bn ${fix_bn} \
                    --use_mux ${use_bn_mux} \
<<<<<<< HEAD
=======
                    --rpn_univ ${rpn_univ} \
>>>>>>> 793eeda709a4483589939795954491531204c768
                    --randomly_chosen_datasets ${randomly_chosen_datasets} \
                    --update_chosen ${update_chosen} \
                    --warmup_steps ${warmup_steps} \
                    --backward_together ${backward_together} \
<<<<<<< HEAD
                    # --r ${resume} \
                    # --checksession ${checksession} \
                    # --checkpoint ${CHECKPOINT} \
                    # --checkepoch ${checkepoch} \
=======
                    --r ${resume} \
                    --checksession ${checksession} \
                    --checkpoint ${CHECKPOINT} \
                    --checkepoch ${checkepoch} \
>>>>>>> 793eeda709a4483589939795954491531204c768
