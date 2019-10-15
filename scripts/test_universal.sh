rm /home/Xwang/HeadNode-1/universal_model/lib/model/faster_rcnn/faster_rcnn_uni.pyc
rm /home/Xwang/HeadNode-1/universal_model/lib/model/faster_rcnn/dataset_attention_module.pyc
rm /home/Xwang/HeadNode-1/universal_model/lib/model/faster_rcnn/SEResNet_Data_Attention.pyc
rm /home/Xwang/HeadNode-1/universal_model/lib/model/rpn/rpn_universal.pyc
rm /home/Xwang/HeadNode-1/universal_model/lib/datasets/datasets_info.pyc
BATCH_SIZE=2
net=data_att_res50
WORKER_NUMBER=2
LEARNING_RATE=0.001
DECAY_STEP=5
GPU_ID=3
SESSION=11083
CHECKPOINT=5720 #SESSION=11086 #CHECKPOINT=5720

# datasets=caltech
# datasets=citypersons
datasets=coco
datasets=comic
datasets=watercolor
datasets=deeplesion
datasets=clipart
datasets=dota
datasets=KITTIVOC
datasets=pascal_voc_0712
datasets=widerface
datasets=Kitchen
datasets=LISA

finetuneBN_DS=0
finetuneBN_linear=0
fa_conv_num=0

random_resize=False
fix_bn=True
use_bn_mux=False
DATA_DIR=/home/Xwang/HeadNode-1/universal_model_/data
num_se=3
less_blocks=True
rpn_univ=False

EPOCH=11
CUDA_VISIBLE_DEVICES=${GPU_ID} python test_universal.py \
                    --dataset ${datasets} --net ${net} \
                    --checksession ${SESSION} \
                    --checkepoch ${EPOCH} \
                    --checkpoint ${CHECKPOINT} \
                    --cuda --mGPUs \
                    --DATA_DIR ${DATA_DIR} \
                    --random_resize ${random_resize} \
                    --num_se ${num_se} \
                    --fix_bn ${fix_bn} \
                    --rpn_univ ${rpn_univ} \
                    --less_blocks ${less_blocks} \
                    --use_mux ${use_bn_mux} \
                    --finetuneBN_DS ${finetuneBN_DS} \
                    --finetuneBN_linear ${finetuneBN_linear} \
                    --fa_conv_num ${fa_conv_num} \

EPOCH=12
CUDA_VISIBLE_DEVICES=${GPU_ID} python test_universal.py \
                    --dataset ${datasets} --net ${net} \
                    --checksession ${SESSION} \
                    --checkepoch ${EPOCH} \
                    --checkpoint ${CHECKPOINT} \
                    --cuda --mGPUs \
                    --DATA_DIR ${DATA_DIR} \
                    --random_resize ${random_resize} \
                    --num_se ${num_se} \
                    --fix_bn ${fix_bn} \
                    --rpn_univ ${rpn_univ} \
                    --less_blocks ${less_blocks} \
                    --use_mux ${use_bn_mux} \
                    --finetuneBN_DS ${finetuneBN_DS} \
                    --finetuneBN_linear ${finetuneBN_linear} \
                    --fa_conv_num ${fa_conv_num} \

EPOCH=13
CUDA_VISIBLE_DEVICES=${GPU_ID} python test_universal.py \
                    --dataset ${datasets} --net ${net} \
                    --checksession ${SESSION} \
                    --checkepoch ${EPOCH} \
                    --checkpoint ${CHECKPOINT} \
                    --cuda --mGPUs \
                    --DATA_DIR ${DATA_DIR} \
                    --random_resize ${random_resize} \
                    --num_se ${num_se} \
                    --fix_bn ${fix_bn} \
                    --rpn_univ ${rpn_univ} \
                    --less_blocks ${less_blocks} \
                    --use_mux ${use_bn_mux} \
                    --finetuneBN_DS ${finetuneBN_DS} \
                    --finetuneBN_linear ${finetuneBN_linear} \
                    --fa_conv_num ${fa_conv_num} \

EPOCH=14
CUDA_VISIBLE_DEVICES=${GPU_ID} python test_universal.py \
                    --dataset ${datasets} --net ${net} \
                    --checksession ${SESSION} \
                    --checkepoch ${EPOCH} \
                    --checkpoint ${CHECKPOINT} \
                    --cuda --mGPUs \
                    --DATA_DIR ${DATA_DIR} \
                    --random_resize ${random_resize} \
                    --num_se ${num_se} \
                    --fix_bn ${fix_bn} \
                    --rpn_univ ${rpn_univ} \
                    --less_blocks ${less_blocks} \
                    --use_mux ${use_bn_mux} \
                    --finetuneBN_DS ${finetuneBN_DS} \
                    --finetuneBN_linear ${finetuneBN_linear} \
                    --fa_conv_num ${fa_conv_num} \

EPOCH=16
CUDA_VISIBLE_DEVICES=${GPU_ID} python test_universal.py \
                    --dataset ${datasets} --net ${net} \
                    --checksession ${SESSION} \
                    --checkepoch ${EPOCH} \
                    --checkpoint ${CHECKPOINT} \
                    --cuda --mGPUs \
                    --DATA_DIR ${DATA_DIR} \
                    --random_resize ${random_resize} \
                    --num_se ${num_se} \
                    --fix_bn ${fix_bn} \
                    --rpn_univ ${rpn_univ} \
                    --less_blocks ${less_blocks} \
                    --use_mux ${use_bn_mux} \
                    --finetuneBN_DS ${finetuneBN_DS} \
                    --finetuneBN_linear ${finetuneBN_linear} \
                    --fa_conv_num ${fa_conv_num} \