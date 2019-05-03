#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python train.py \
	--epochs=51 \
	--batch-size=16 \
	--cfg="cfg/yolov3-dfsign.cfg" \
	--img-size=416 \

