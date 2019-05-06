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
    python train_ssd.py \
        --network="vgg16_atrous" \
        --data-shape=300 \
        --batch-size=32 \
        --dataset="tt100k" \
        --dataset_root="${TT100K_ROOT}" \
        --num-workers=16 \
        --gpus="0" \
        --epochs=240 \
        --resume="weights/tt100k_0225.params" \
        --start-epoch=226 \
        --lr=1e-3 \
        --lr-decay-epoch="160,200" \
        --momentum=0.9 \
        --wd=5e-4 \
        --val=0
elif [ 2 == $FLAG ]
then
    echo "====test===="
    python test_ssd.py \
        --network="vgg16_atrous" \
        --dataset="tt100k" \
        --dataset_root="${TT100K_ROOT}" \
        --pretrained="weights/tt100k_0225.params"
elif [ 3 == $FLAG ]
then
    echo "====eval===="
    python eval.py \
        --network="RFB" \
        --trained_model="weights/VOC.pth" \
        --voc_root="${VOC_ROOT}" 
elif [ 3 == $FLAG ]
then
    echo "====test 2===="
    python test_ssd.py \
        --network="vgg16_atrous" \
        --dataset="voc" \
        --dataset_root="${VOC_ROOT}"
else
    echo "error"
fi

# Run inference with the exported checkpoint.
# Please refer to the provided deeplab_demo.ipynb for an example.
