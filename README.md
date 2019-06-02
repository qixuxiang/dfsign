### requirements
```
python=3.7
    numpy
    pandas
    opencv-python
    pytorch >= 1.0.0
    matplotlib
    pycocotools
    tqdm
    cython
    cffi
    opencv-python >= 4.0
    scipy
    msgpack
    easydict
    matplotlib
    pyyaml
    tensorboardX
```

### 数据准备
数据目录为`~/data/dfsign`，下载zip文件和label文件，解压zip到test和train目录
```
~/data
├── dfsign
│   ├── test
│   ├── train
│   ├── Test_fix.zip
│   ├── Train_fix.zip
│   ├── train_label_fix.csv
```

### 代码准备
代码需要放在特定目录`$WORKDIR = ~/working/dfsign`

链接数据
``` bash
ln -s ~/data $WORKDIR/mmdetection
```

mmdetection extensions
``` bash
cd $WORKDIR/mmdetection
./compile.sh
python setup.py develop
```

### 模型权重准备
1. 将deeplab模型放在`$WORKDIR/pytorch-deeplab-xception/run/dfsign`
2. 将detection模型放在`WORKDIR/mmdetection/dfsign/work_dirs`

### 生成训练数据
``` bash
cd $WORKDIR/tools
# generate segmentation dataset
python convert2voc.py
# generate detection trainset
python generate_train_chip.py
```

### 训练
pass

### 测试
``` bash
cd $WORKDIR/pytorch-deeplab-xception
# run deeplab
./test.sh

cd $WORKDIR/tools
# crop seg results
python generate_mask_chip.py

cd $WORKDIR/mmdetection/dfsign
# run detect model_1 on images from seg results
python detect.py cascade_rcnn_x101_64x4d_fpn.py work_dirs/cascade_rcnn_x101_64x4d_fpn_1x/9954.pth --chip

cd $WORKDIR/tools
# crop detect results
python generate_detect_chip.py

cd $WORKDIR/mmdetection/dfsign
# run detect model_1 on images from detect results
python detect.py cascade_rcnn_x101_64x4d_fpn.py work_dirs/cascade_rcnn_x101_64x4d_fpn_1x/9954.pth
cd $WORKDIR/tools
# predict_1
python dfsign_submit.py predict_1

cd $WORKDIR/mmdetection/dfsign
# 修改cascade_rcnn_x101_64x4d_fpn_1x.py中25行anchor_ratios为[0.5, 1.0, 2.0]
# run detect model_2 on images from detect results
python detect.py cascade_rcnn_x101_64x4d_fpn.py work_dirs/cascade_rcnn_x101_64x4d_fpn_1x/9946.pth
cd $WORKDIR/tools
# predict_2
python dfsign_submit.py predict_2

# ensemble
python dfsign_ensemble
# 生成的predict.csv为最终结果
```
