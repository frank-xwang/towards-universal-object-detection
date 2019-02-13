rm /home/Xwang/HeadNode-1/universal_model/lib/model/faster_rcnn/faster_rcnn_uni.pyc
rm /home/Xwang/HeadNode-1/universal_model/lib/model/faster_rcnn/dataset_attention_module.pyc
rm /home/Xwang/HeadNode-1/universal_model/lib/model/faster_rcnn/SEResNet_Data_Attention.pyc
rm /home/Xwang/HeadNode-1/universal_model/lib/model/rpn/rpn_universal.pyc
BATCH_SIZE=12
net=data_att_res50
WORKER_NUMBER=1
LEARNING_RATE=0.01
DECAY_STEP=10
GPU_ID=0,1,2,3
checkepoch=1
SAVE_SESSION=1
checksession=1
CHECKPOINT=171
epochs=20
resume=True
backward_together=0 # 0: independent; 1: together
USE_FLIPPED=1 # choose 1 for using flipped images, 0 for don't

datasets=universal
DATA_DIR=/home/Xwang/HeadNode-1/universal_model_/data

random_resize=True
fix_bn=True
use_bn_mux=False
domain_pred=True
domain_pred_weight=0.5

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
                    --fix_bn ${fix_bn} \
                    --use_mux ${use_bn_mux} \
                    --domain_pred ${domain_pred} \
                    --domain_pred_weight ${domain_pred_weight} \
                    --backward_together ${backward_together}
                    --r ${resume} \
                    --checksession ${checksession} \
                    --checkepoch ${checkepoch} \
                    --checkpoint ${CHECKPOINT}
