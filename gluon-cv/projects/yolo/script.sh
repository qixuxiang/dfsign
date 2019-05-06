#!/bin/bash
# Exit immediately if a command exits with a non-zero status.
set -e

FLAG=$1

# Set up the working directories.
VOC_ROOT="${HOME}/data/VOCdevkit"
COCO_ROOT="${HOME}/data/COCO"
TT100K_ROOT="${HOME}/data/TT100K/TT100K_chip_voc"

echo $FLAG
if [ 1 == $FLAG ] 
then
    echo "====train===="
    MXNET_CUDNN_AUTOTUNE_DEFAULT=0 python train_yolo3.py \
        --network="darknet53" \
        --batch-size=8 \
        --dataset="tt100k" \
        --dataset_root="${TT100K_ROOT}" \
        --num-workers=8 \
        --gpus="0,1" \
        --epochs=200 \
        --resume="" \
        --start-epoch=0 \
        --lr=1e-3 \
        --lr-decay-epoch="160" \
        --momentum=0.9 \
        --wd=5e-4 \
        --val=0 \
        --warmup-epochs=4 \
        --syncbn \
	--label-smooth
elif [ 2 == $FLAG ]
then
    echo "====test===="
    python test_yolo.py \
        --network="darknet53" \
        --dataset="tt100k" \
        --dataset_root="${TT100K_ROOT}" \
        --pretrained="weights/yolo3_darknet53_custom_0199.params"
else
    echo "error"
fi

# Run inference with the exported checkpoint.
# Please refer to the provided deeplab_demo.ipynb for an example.
