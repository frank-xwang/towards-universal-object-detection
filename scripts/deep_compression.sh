BATCH_SIZE=6
net=res50
WORKER_NUMBER=6
LEARNING_RATE=0.001
DECAY_STEP=9
GPU_ID=2,3
checkepoch=10
SESSION=2
CHECKPOINT=1869

#number of epochs to train
epochs=16
resume=True
datasets=KAISTVOC
datasets=caltech
datasets=pascal_voc
datasets=citypersons
datasets=pascal_voc_0712
datasets=widerface
datasets=KITTIVOC

CUDA_VISIBLE_DEVICES=${GPU_ID} python deep_compression.py \
                    --dataset ${datasets} --net ${net} \
                    --bs ${BATCH_SIZE} --nw ${WORKER_NUMBER} \
                    --lr ${LEARNING_RATE} --lr_decay_step ${DECAY_STEP} \
                    --cuda --mGPUs \
                    --s ${SESSION} \
                    --epochs ${epochs} \
                    --checksession ${SESSION} \
                    --checkepoch ${checkepoch} \
                    --checkpoint ${CHECKPOINT}
                    --r ${resume} 
