def train_params(net, aim):
    if aim == 'fine_tune':
        if net == 'vgg16_bn_finetune':
            #number of epochs to train
            PRE_datasets=pascal_voc_0712
            PRE_epoch=11
            PRE_ses=10
            PRE_checkpoint=8274
            #datasets=caltech
            datasets=KITTIVOC
            #datasets=pascal_voc_0712
            #datasets=coco
            #datasets=universal
            datasets=widerface

            return BATCH_SIZE, checkepoch, SESSION, DECAY_STEP, 
        if net == 'res18_fine_tune':
            BATCH_SIZE=4
            WORKER_NUMBER=8
            LEARNING_RATE=0.004
            DECAY_STEP=8
            GPU_ID=4,5,6,8
            GPU_ID=5,6,7,8
            GPU_ID=2,7
            checkepoch=2
            SESSION=12
            CHECKPOINT=6437
            #number of epochs to train
            epochs=14
            resume=True
            backward_together=False
            PRE_datasets=pascal_voc_0712
            PRE_epoch=11
            PRE_ses=10
            PRE_checkpoint=8274
            #datasets=caltech
            datasets=KITTIVOC
            #datasets=pascal_voc_0712
            #datasets=coco
            #datasets=universal
            datasets=widerface