BATCH_SIZE=2
net=data_att_res50
WORKER_NUMBER=8
LEARNING_RATE=0.001
DECAY_STEP=5
GPU_ID=1
EPOCH=12
SESSION=154
CHECKPOINT=3813

# datasets=caltech
# datasets=caltech
# datasets=citypersons
# datasets=pascal_voc
# datasets=citypersons
# datasets=KAISTVOC

# datasets=KITTIVOC
# datasets=widerface
# datasets=pascal_voc_0712
datasets=comic
datasets=watercolor
datasets=clipart
datasets=coco
datasets=dota
datasets=pascal_voc_0712
datasets=widerface
datasets=KITTIVOC
datasets=Kitchen

finetuneBN_DS=0
finetuneBN_linear=0
fa_conv_num=0

random_resize=False
fix_bn=True
use_bn_mux=False

CUDA_VISIBLE_DEVICES=${GPU_ID} python test_universal.py \
                    --dataset ${datasets} --net ${net} \
                    --checksession ${SESSION} \
                    --checkepoch ${EPOCH} \
                    --checkpoint ${CHECKPOINT} \
                    --cuda --mGPUs \
                    --random_resize ${random_resize} \
                    --fix_bn ${fix_bn} \
                    --use_mux ${use_bn_mux} \
                    --finetuneBN_DS ${finetuneBN_DS} \
                    --finetuneBN_linear ${finetuneBN_linear} \
                    --fa_conv_num ${fa_conv_num} \

# EPOCH=11
# CUDA_VISIBLE_DEVICES=${GPU_ID} python test_universal.py \
#                     --dataset ${datasets} --net ${net} \
#                     --checksession ${SESSION} \
#                     --checkepoch ${EPOCH} \
#                     --checkpoint ${CHECKPOINT} \
#                     --cuda --mGPUs \
#                     --random_resize ${random_resize} \
#                     --fix_bn ${fix_bn} \
#                     --use_mux ${use_bn_mux} \
#                     --finetuneBN_DS ${finetuneBN_DS} \
#                     --finetuneBN_linear ${finetuneBN_linear} \
#                     --fa_conv_num ${fa_conv_num} \

# EPOCH=12
# CUDA_VISIBLE_DEVICES=${GPU_ID} python test_universal.py \
#                     --dataset ${datasets} --net ${net} \
#                     --checksession ${SESSION} \
#                     --checkepoch ${EPOCH} \
#                     --checkpoint ${CHECKPOINT} \
#                     --cuda --mGPUs \
#                     --random_resize ${random_resize} \
#                     --fix_bn ${fix_bn} \
#                     --use_mux ${use_bn_mux} \
#                     --finetuneBN_DS ${finetuneBN_DS} \
#                     --finetuneBN_linear ${finetuneBN_linear} \
#                     --fa_conv_num ${fa_conv_num} \

# EPOCH=13
# CUDA_VISIBLE_DEVICES=${GPU_ID} python test_universal.py \
#                     --dataset ${datasets} --net ${net} \
#                     --checksession ${SESSION} \
#                     --checkepoch ${EPOCH} \
#                     --checkpoint ${CHECKPOINT} \
#                     --cuda --mGPUs \
#                     --random_resize ${random_resize} \
#                     --fix_bn ${fix_bn} \
#                     --use_mux ${use_bn_mux} \
#                     --finetuneBN_DS ${finetuneBN_DS} \
#                     --finetuneBN_linear ${finetuneBN_linear} \
#                     --fa_conv_num ${fa_conv_num} \

# EPOCH=14
# CUDA_VISIBLE_DEVICES=${GPU_ID} python test_universal.py \
#                     --dataset ${datasets} --net ${net} \
#                     --checksession ${SESSION} \
#                     --checkepoch ${EPOCH} \
#                     --checkpoint ${CHECKPOINT} \
#                     --cuda --mGPUs \
#                     --random_resize ${random_resize} \
#                     --fix_bn ${fix_bn} \
#                     --use_mux ${use_bn_mux} \
#                     --finetuneBN_DS ${finetuneBN_DS} \
#                     --finetuneBN_linear ${finetuneBN_linear} \
#                     --fa_conv_num ${fa_conv_num} \

# EPOCH=15
# CUDA_VISIBLE_DEVICES=${GPU_ID} python test_universal.py \
#                     --dataset ${datasets} --net ${net} \
#                     --checksession ${SESSION} \
#                     --checkepoch ${EPOCH} \
#                     --checkpoint ${CHECKPOINT} \
#                     --cuda --mGPUs \
#                     --random_resize ${random_resize} \
#                     --fix_bn ${fix_bn} \
#                     --use_mux ${use_bn_mux} \
#                     --finetuneBN_DS ${finetuneBN_DS} \
#                     --finetuneBN_linear ${finetuneBN_linear} \
#                     --fa_conv_num ${fa_conv_num} \