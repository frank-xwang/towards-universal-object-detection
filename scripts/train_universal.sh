GPU_ID=0,1,2,3,4,5,6,7
BATCH_SIZE=16
net=da_res50
WORKER_NUMBER=2
LEARNING_RATE=0.01 # lr=0.01 for BATCH_SIZE=16
DECAY_STEP=10
SAVE_SESSION=11
epochs=14
backward_together=0 # 0: independent; 1: together
USE_FLIPPED=1 # choose 1 for using flipped images, 0 for don't
datasets=universal
DATA_DIR=data
random_resize=True
fix_bn=True
use_bn_mux=False
update_chosen=False
randomly_chosen_datasets=True
warmup_steps=0
num_adapters=11
less_blocks=False
datasets_list='KITTI widerface pascal_voc_0712 Kitchen LISA deeplesion coco clipart comic watercolor dota'
# datasets_list='KITTI widerface pascal_voc_0712 Kitchen LISA'
# resume=True
# checkepoch=7
# checksession=11083
# CHECKPOINT=5720

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
                    --num_adapters ${num_adapters} \
                    --less_blocks ${less_blocks} \
                    --fix_bn ${fix_bn} \
                    --use_mux ${use_bn_mux} \
                    --randomly_chosen_datasets ${randomly_chosen_datasets} \
                    --update_chosen ${update_chosen} \
                    --warmup_steps ${warmup_steps} \
                    --backward_together ${backward_together} \
                    --datasets_list ${datasets_list}
                    # --r ${resume} \
                    # --checksession ${checksession} \
                    # --checkpoint ${CHECKPOINT} \
                    # --checkepoch ${checkepoch} \