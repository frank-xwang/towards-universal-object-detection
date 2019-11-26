BATCH_SIZE=4
net=se_res50
WORKER_NUMBER=8
LEARNING_RATE=0.001
DECAY_STEP=5
GPU_ID=0
EPOCH=12
SESSION=133
CHECKPOINT=249

datasets=KAISTVOC
test_model=KAISTVOC

#test_model=cross_domain
datasets=citypersons
test_model=citypersons
datasets=caltech
test_model=caltech
datasets=pascal_voc
test_model=pascal_voc
test_model=coco
datasets=coco
datasets=dota
test_model=dota
datasets=KITTIVOC
test_model=KITTIVOC

datasets=deeplesion
test_model=deeplesion
datasets=widerface
test_model=widerface
datasets=LISA
test_model=LISA
datasets=Kitchen
test_model=Kitchen
datasets=pascal_voc_0712
test_model=pascal_voc_0712
test_model=watercolor
datasets=comic
datasets=clipart
datasets=watercolor

<<<<<<< HEAD
DATA_DIR=data
=======
DATA_DIR=/home/Xwang/HeadNode-1/universal_model_/data
#/home/Xwang/HeadNode-1/universal_model_/models/se_res50/Kitchen/faster_rcnn_128_14_587.pth
>>>>>>> 793eeda709a4483589939795954491531204c768

EPOCH=12
CUDA_VISIBLE_DEVICES=${GPU_ID} python test_net.py \
                    --dataset ${datasets} --net ${net} \
                    --checksession ${SESSION} \
                    --checkepoch ${EPOCH} \
                    --checkpoint ${CHECKPOINT} \
                    --DATA_DIR ${DATA_DIR} \
                    --TEST_MODEL ${test_model} \
                    --cuda --mGPUs \
                    # --TEST_MODEL ${test_model} \

EPOCH=14
CUDA_VISIBLE_DEVICES=${GPU_ID} python test_net.py \
                    --dataset ${datasets} --net ${net} \
                    --checksession ${SESSION} \
                    --checkepoch ${EPOCH} \
                    --checkpoint ${CHECKPOINT} \
                    --TEST_MODEL ${test_model} \
                    --cuda --mGPUs \
                    # --TEST_MODEL ${test_model} \