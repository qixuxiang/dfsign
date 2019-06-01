 CUDA_VISIBLE_DEVICES=0,1 python trainval_net.py \
                    --dataset dfsign --net res101 \
                    --bs 4 --nw 4 --mGPUs \
                    --lr 0.002 --lr_decay_step 20 \
                    --cuda
