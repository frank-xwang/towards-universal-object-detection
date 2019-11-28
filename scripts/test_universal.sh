GPU_ID=0
batch_size=1
net=da_res50
DATA_DIR=data
num_adapters=8
less_blocks=True # False

### Arguments for checkpoints
EPOCH=14
SESSION=8
CHECKPOINT=13331

### Choose the datasest to test
datasets_test=KITTI
# datasets_list='KITTI widerface pascal_voc_0712 Kitchen LISA'
datasets_list='LISA pascal_voc_0712 Kitchen coco clipart watercolor comic widerface dota deeplesion KITTI'

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${GPU_ID} \
python test_universal.py \
--dataset ${datasets_test} --net ${net} \
--checksession ${SESSION} \
--checkepoch ${EPOCH} \
--checkpoint ${CHECKPOINT} \
--cuda --mGPUs \
--DATA_DIR ${DATA_DIR} \
--num_adapters ${num_adapters} \
--less_blocks ${less_blocks} \
--datasets_list ${datasets_list} \
--bs ${batch_size}
