#!/bin/bash
# Exit immediately if a command exits with a non-zero status.
set -e

FLAG=$1

# Set up the working directories.
VOC_ROOT="${HOME}/data/VOCdevkit"
COCO_ROOT="${HOME}/data/COCO"
TT100K_ROOT="${HOME}/data/TT100K/TT100K_voc"

echo $FLAG
if [ 1 == $FLAG ] 
then
    echo "====train===="
    python train.py \
        --model="deeplab" \
        --backbone="resnet50" \
        --dataset="tt100k_region" \
        --dataset_root="${TT100K_ROOT}" \
        --base-size=520 \
        --crop-size=480 \
        --batch-size=4 \
        --workers=8 \
        --epochs=35 \
        --resume="weights/tt100k_region_deeplab_resnet50_0025.params" \
        --start-epoch=26 \
        --lr=1e-3 \
        --momentum=0.9 \
        --weight-decay=1e-5 \
        --aux-weight=0.2 \
        --syncbn \
        --no-val \
        --checkname="resnet50"
elif [ 2 == $FLAG ]
then
    echo "====test===="
    python test.py \
        --model="deeplab" \
        --backbone="resnet50" \
        --dataset="tt100k_region" \
        --dataset_root="${TT100K_ROOT}" \
        --crop-size=640 \
        --resume="weights/tt100k_region_deeplab_resnet50_0025.params" \
        --syncbn \
        --no-val \
        --ngpus=1 \
        --checkname="resnet101"
fi

# Run inference with the exported checkpoint.
# Please refer to the provided deeplab_demo.ipynb for an example.
