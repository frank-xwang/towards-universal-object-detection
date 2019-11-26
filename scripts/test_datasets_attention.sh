BATCH_SIZE=2
net=data_att_res50
WORKER_NUMBER=8
LEARNING_RATE=0.01
DECAY_STEP=5
GPU_ID=0
EPOCH=10
SESSION=35
CHECKPOINT=2198

datasets=watercolor
datasets=clipart
datasets=comic
datasets=coco
datasets=Kitchen
datasets=dota
datasets=KITTIVOC
datasets=pascal_voc_0712
datasets=widerface

finetuneBN_DS=0
finetuneBN_linear=0
fa_conv_num=0

EPOCH=10
CUDA_VISIBLE_DEVICES=${GPU_ID} python test_dataset_attention.py \
                    --dataset ${datasets} --net ${net} \
                    --checksession ${SESSION} \
                    --checkepoch ${EPOCH} \
                    --checkpoint ${CHECKPOINT} \
                    --cuda --mGPUs \
                    --finetuneBN_DS ${finetuneBN_DS} \
                    --finetuneBN_linear ${finetuneBN_linear} \
                    --fa_conv_num ${fa_conv_num} \

EPOCH=11
CUDA_VISIBLE_DEVICES=${GPU_ID} python test_dataset_attention.py \
                    --dataset ${datasets} --net ${net} \
                    --checksession ${SESSION} \
                    --checkepoch ${EPOCH} \
                    --checkpoint ${CHECKPOINT} \
                    --cuda --mGPUs \
                    --finetuneBN_DS ${finetuneBN_DS} \
                    --finetuneBN_linear ${finetuneBN_linear} \
                    --fa_conv_num ${fa_conv_num} \

EPOCH=12
CUDA_VISIBLE_DEVICES=${GPU_ID} python test_dataset_attention.py \
                    --dataset ${datasets} --net ${net} \
                    --checksession ${SESSION} \
                    --checkepoch ${EPOCH} \
                    --checkpoint ${CHECKPOINT} \
                    --cuda --mGPUs \
                    --finetuneBN_DS ${finetuneBN_DS} \
                    --finetuneBN_linear ${finetuneBN_linear} \
                    --fa_conv_num ${fa_conv_num} \

EPOCH=13
CUDA_VISIBLE_DEVICES=${GPU_ID} python test_dataset_attention.py \
                    --dataset ${datasets} --net ${net} \
                    --checksession ${SESSION} \
                    --checkepoch ${EPOCH} \
                    --checkpoint ${CHECKPOINT} \
                    --cuda --mGPUs \
                    --finetuneBN_DS ${finetuneBN_DS} \
                    --finetuneBN_linear ${finetuneBN_linear} \
                    --fa_conv_num ${fa_conv_num} \

EPOCH=14
CUDA_VISIBLE_DEVICES=${GPU_ID} python test_dataset_attention.py \
                    --dataset ${datasets} --net ${net} \
                    --checksession ${SESSION} \
                    --checkepoch ${EPOCH} \
                    --checkpoint ${CHECKPOINT} \
                    --cuda --mGPUs \
                    --finetuneBN_DS ${finetuneBN_DS} \
                    --finetuneBN_linear ${finetuneBN_linear} \
                    --fa_conv_num ${fa_conv_num} \

EPOCH=15
CUDA_VISIBLE_DEVICES=${GPU_ID} python test_dataset_attention.py \
                    --dataset ${datasets} --net ${net} \
                    --checksession ${SESSION} \
                    --checkepoch ${EPOCH} \
                    --checkpoint ${CHECKPOINT} \
                    --cuda --mGPUs \
                    --finetuneBN_DS ${finetuneBN_DS} \
                    --finetuneBN_linear ${finetuneBN_linear} \
                    --fa_conv_num ${fa_conv_num} \