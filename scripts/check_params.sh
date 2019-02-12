BATCH_SIZE=8
net=vgg16
WORKER_NUMBER=8
LEARNING_RATE=0.001
DECAY_STEP=5
GPU_ID=5
EPOCH=13
SESSION=1
CHECKPOINT=5604
datasets=caltech
#datasets=KITTIVOC
#datasets=pascal_voc
#test_model=KITTIVOC
#test_model=universal
test_model=caltech
#test_model=pascal_voc
CUDA_VISIBLE_DEVICES=${GPU_ID} python check_params.py \
                    --dataset ${datasets} --net ${net} \
                    --checksession ${SESSION} \
                    --checkepoch ${EPOCH} \
                    --checkpoint ${CHECKPOINT} \
                    --TEST_MODEL ${test_model} \
                    --cuda --mGPUs