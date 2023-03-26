# AVM

## 环境配置

1. 使用 MIM 安装 MMCV 和 MMDetection

```shell
pip install -U openmim
mim install mmcv-full
mim install mmdet\<3.0.0
```

2. 安装 MMSegmentation

```shell
pip install mmsegmentation
```

3. 安装 MMRotate

```shell
git clone https://github.com/Yuppie898988/AVM.git
cd AVM
pip install -v -e .
```

## 数据准备

AVM原始数据集目录需调整为如下结构

```python
AVM
 └─ data
     └─ avm
         ├─ images
         │   ├─ b2_left
         │   │   └─ avm
         │   │       └─ *.jpg
         │   ├─ b2_right
         │   ├─ b2_to_b3
         │   └─ b3_to_b2
         ├─ det_annotations
         │   ├─ b2_left
         │   │   └─ *.json
         │   ├─ b2_right
         │   ├─ b2_to_b3
         │   └─ b3_to_b2
         └─ mask
             ├─ b2_left
             │   └─ *.png
             ├─ b2_right
             ├─ b2_to_b3
             └─ b3_to_b2
```

通过`create_data.py`将数据集转换为指定格式

```shell
python tools/create_data.py --avm-dir ./data/avm --output-dir ./data/avm_seg_det
```

转换后格式为

```python
AVM
 └─ data
     ├─ avm
     │   └─ ...
     └─ avm_seg_det
         └─ train
             ├─ annfiles
             │   └─ *.txt
             ├─ images
             │   └─ *.png
             └─ segments
                 └─ *.png
```

## 训练

```shell
./tools/dist_train.sh configs/avm/seg_rotated_retinanet_obb_r50_fpn_1x_avm_le90.py 8 --work-dir ${YOUR_WORK_DIR}
```

## 推理

```shell
python tools/inference.py --config configs/avm/seg_rotated_retinanet_obb_r50_fpn_1x_avm_le90.py --checkpoint ${CHECKPOINT_FILE} --image_path ${IMAGE_FILE} --save_dir ${SAVE_DIR_PATH}
```

## 样例

|Input|Detection|Segmentation|
|-|-|-|
|<img src=demo/avm_demo.png width=300 height=300 />|<img src=demo/avm_demo_detect.png width=300 height=300 />|<img src=demo/avm_demo_seg.png width=300 height=300 />|
