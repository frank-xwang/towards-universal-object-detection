BATCH_SIZE=24
net=data_att_res50
# net=data_att_res50
WORKER_NUMBER=1
LEARNING_RATE=0.01
DECAY_STEP=10
GPU_ID=9,6,7,8
GPU_ID=0,1,2,3
# GPU_ID=6,7
# GPU_ID=0,1,2,3,7
# GPU_ID=4
GPU_ID=0,1,2,3,4,5,6,7
checkepoch=1
SAVE_SESSION=1
checksession=1
CHECKPOINT=171
epochs=20
resume=True
backward_together=0 # 0: independent; 1: together
USE_FLIPPED=0 # choose 1 for using flipped images, 0 for don't

datasets=universal

DATA_DIR=/gpu7_ssd/xuw080/univ_data/
DATA_DIR=/gpu2_data/xuw080/data/
DATA_DIR=/gpu3_data/xuw080/data/
DATA_DIR=/data4/xuw080/universal_model/data/
DATA_DIR=/gpu7_ssd/xuw080/univ_data/
DATA_DIR=/gpu6_ssd/xuw080/univ_data/
DATA_DIR=/gpu5_ssd/xuw080/data/
DATA_DIR=/mnt/local_mnt/sda/xudong/univ_data

random_resize=True
fix_bn=True
use_bn_mux=False

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
                    --fix_bn ${fix_bn} \
                    --use_mux ${use_bn_mux} \
                    --backward_together ${backward_together}
                    --r ${resume} \
                    --checksession ${checksession} \
                    --checkepoch ${checkepoch} \
                    --checkpoint ${CHECKPOINT}
