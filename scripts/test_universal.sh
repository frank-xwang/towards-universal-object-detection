BATCH_SIZE=2
net=da_res50
WORKER_NUMBER=2
LEARNING_RATE=0.001
DECAY_STEP=5
GPU_ID=8
SESSION=1100
CHECKPOINT=13331

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
use_bn_mux=False
DATA_DIR=data
num_se=11
less_blocks=False

EPOCH=14
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${GPU_ID} \
python test_universal.py \
--dataset ${datasets} --net ${net} \
--checksession ${SESSION} \
--checkepoch ${EPOCH} \
--checkpoint ${CHECKPOINT} \
--cuda --mGPUs \
--DATA_DIR ${DATA_DIR} \
--random_resize ${random_resize} \
--num_se ${num_se} \
--less_blocks ${less_blocks}