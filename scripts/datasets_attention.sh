BATCH_SIZE=16
net=data_att_res50
WORKER_NUMBER=1
LEARNING_RATE=0.01
DECAY_STEP=8
#GPU_ID=5,6,7,8
GPU_ID=1,2,4,7
GPU_ID=2,3,4,5
GPU_ID=6,7,8,9
GPU_ID=4,5,6,7
checkepoch=6
SESSION=40
SAVE_session=40
CHECKPOINT=2198
#number of epochs to train
epochs=20
resume=True
backward_together=0 # 0: independent; 1: together
USE_FLIPPED=1 # choose 1 for using flipped images, 0 for don't
datasets=universal
#DATA_DIR=/gpu2_data/xuw080/data/
DATA_DIR=/gpu5_ssd/xuw080/data/
DATA_DIR=/gpu5_data/xuw080/data/
DATA_DIR=/gpu7_ssd/xuw080/univ_data/
DATA_DIR=/gpu4_data/xuw080/data/
DATA_DIR=/data4/xuw080/universal_model/data/

random_resize=True
cycle_iter=True

CUDA_VISIBLE_DEVICES=${GPU_ID} python datasets_attention_train.py \
                    --dataset ${datasets} --net ${net} \
                    --bs ${BATCH_SIZE} --nw ${WORKER_NUMBER} \
                    --lr ${LEARNING_RATE} --lr_decay_step ${DECAY_STEP} \
                    --USE_FLIPPED ${USE_FLIPPED} \
                    --cuda --mGPUs \
                    --s ${SAVE_session} \
                    --epochs ${epochs} \
                    --cycle_iter ${cycle_iter} \
                    --DATA_DIR ${DATA_DIR} \
                    --random_resize ${random_resize} \
                    --backward_together ${backward_together} \
                    --r ${resume} \
                    --checksession ${SESSION} \
                    --checkepoch ${checkepoch} \
                    --checkpoint ${CHECKPOINT}