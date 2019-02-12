BATCH_SIZE=4
net=se_res50
WORKER_NUMBER=8
LEARNING_RATE=0.001
DECAY_STEP=5
GPU_ID=2
EPOCH=12
SESSION=125
CHECKPOINT=1608

datasets=KAISTVOC
test_model=KAISTVOC

test_model=comic
test_model=clipart
#test_model=cross_domain
datasets=citypersons
test_model=citypersons
datasets=dota
test_model=dota
datasets=caltech
test_model=caltech
test_model=coco
datasets=coco
datasets=watercolor
datasets=comic
datasets=clipart
test_model=cross_domain
datasets=LISA
test_model=LISA
datasets=pascal_voc_0712
test_model=pascal_voc_0712
datasets=KITTIVOC
test_model=KITTIVOC
datasets=deeplesion
test_model=deeplesion
datasets=pascal_voc
test_model=pascal_voc
datasets=widerface
test_model=widerface
datasets=Kitchen
test_model=Kitchen

CUDA_VISIBLE_DEVICES=${GPU_ID} python test_net.py \
                    --dataset ${datasets} --net ${net} \
                    --checksession ${SESSION} \
                    --checkepoch ${EPOCH} \
                    --checkpoint ${CHECKPOINT} \
                    --TEST_MODEL ${test_model} \
                    --cuda --mGPUs

# EPOCH=11
# CUDA_VISIBLE_DEVICES=${GPU_ID} python test_net.py \
#                     --dataset ${datasets} --net ${net} \
#                     --checksession ${SESSION} \
#                     --checkepoch ${EPOCH} \
#                     --checkpoint ${CHECKPOINT} \
#                     --TEST_MODEL ${test_model} \
#                     --cuda --mGPUs \
#                     # --TEST_MODEL ${test_model} \

# EPOCH=12
# CUDA_VISIBLE_DEVICES=${GPU_ID} python test_net.py \
#                     --dataset ${datasets} --net ${net} \
#                     --checksession ${SESSION} \
#                     --checkepoch ${EPOCH} \
#                     --checkpoint ${CHECKPOINT} \
#                     --TEST_MODEL ${test_model} \
#                     --cuda --mGPUs \
#                     # --TEST_MODEL ${test_model} \

# EPOCH=13
# CUDA_VISIBLE_DEVICES=${GPU_ID} python test_net.py \
#                     --dataset ${datasets} --net ${net} \
#                     --checksession ${SESSION} \
#                     --checkepoch ${EPOCH} \
#                     --checkpoint ${CHECKPOINT} \
#                     --TEST_MODEL ${test_model} \
#                     --cuda --mGPUs \
#                     # --TEST_MODEL ${test_model} \

# EPOCH=14
# CUDA_VISIBLE_DEVICES=${GPU_ID} python test_net.py \
#                     --dataset ${datasets} --net ${net} \
#                     --checksession ${SESSION} \
#                     --checkepoch ${EPOCH} \
#                     --checkpoint ${CHECKPOINT} \
#                     --TEST_MODEL ${test_model} \
#                     --cuda --mGPUs \
#                     # --TEST_MODEL ${test_model} \