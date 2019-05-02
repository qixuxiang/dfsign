#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python train.py \
	--epochs=101 \
	--batch-size=16 \
	--cfg="cfg/yolov3-tt100k.cfg" \
	--img-size=416 \

