CUDA_VISIBLE_DEVICES=1 python test.py --backbone resnet --workers 4 --test-batch-size 4 --gpu-ids 0 --dataset tt100k --weight "run/dfsign/resnet-region/experiment_0/model_best.pth.tar"
