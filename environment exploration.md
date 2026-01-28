这里是我们对两个模型的探索

# OMiniTrack

检查以上没有问题后，运行数据处理方法：
```python
python JRDB2019_2d_stitched_converter.py
```
会有下面的场景：
```python
(base) ctt@cq:~/paper3/OmniTrack-main/tools$ python JRDB2019_2d_stitched_converter.py 
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 20/20 [00:07<00:00,  2.79it/s]
Save JRDB_infos_train_v1.2.pkl to /home/ctt/paper3/OmniTrack-main/data/JRDB2019_2d_stitched_anno_pkls
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 27/27 [00:01<00:00, 15.72it/s]
Save JRDB_infos_train_v1.2.pkl to /home/ctt/paper3/OmniTrack-main/data/JRDB2019_2d_stitched_anno_pkls
Save JRDB_infos_test_v1.2.pkl to /home/ctt/paper3/OmniTrack-main/data/JRDB2019_2d_stitched_anno_pkls
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 7/7 [00:02<00:00,  3.07it/s]
Save JRDB_infos_train_v1.2.pkl to /home/ctt/paper3/OmniTrack-main/data/JRDB2019_2d_stitched_anno_pkls
Save JRDB_infos_val_v1.2.pkl to /home/ctt/paper3/OmniTrack-main/data/JRDB2019_2d_stitched_anno_pkls
Save JRDB_infos_test_v1.2.pkl to /home/ctt/paper3/OmniTrack-main/data/JRDB2019_2d_stitched_anno_pkls
```
这一步很顺利，继续。
这里碰到了问题，安装的transformers的版本过高
```python

python anchor_2d_generator.py --ann_file ../data/JRDB2019_2d_stitched_anno_pkls/JRDB_infos_train_v1.2.pkl
```
```python
pip uninstall transformers -y
pip install transformers==4.37.0
```
然后又蹦出来几个包

环境是块难啃的骨头，换一台机器可能就不行。这不，我装了发现numpy 版本不对，欸嘿，torch版本也不对，cuda总该对吧。
选取 cuda 11.8 + torch 2.0.0 + mmvc-full 1.7.1 numpy 1.26.4
```python
cd mmcv-full-1.7.1

# for mmcv-full-1.7.1 GPU version with pip == 24.2
MMCV_WITH_OPS=1 pip install -e .
cd ..
```
并且在安装完成后，要在这个步骤，也就是再下面这个步骤执行之前，加上一个文件
```python
vi pyproject.toml
```
插入下面的内容
```python
[build-system]
requires = [
    "setuptools>=64",
    "wheel",
    "torch==2.0.0"
]
build-backend = "setuptools.build_meta"

[project]
name = "deformable_aggregation_ext"
version = "0.0.0"
description = "Deformable Aggregation Extension for MMDetection3D"
requires-python = ">=3.7"

[tool.setuptools]
zip-safe = false

[tool.setuptools.packages.find]
where = ["."]
```
等这个文件搞好之后，就可以执行了
```python
pip install -e . --config-settings editable_mode=compat
```
这个比下面的方式好用。实测，没有bug
```python
# Compile the deformable_aggregation CUDA op
cd projects/mmdet3d_plugin/ops
python3 setup.py develop
cd ../../../

# Compile the jrdb_toolkit nms
cd jrdb_toolkit/detection_eval
python3 setup.py develop
cd ../../
###
```

然后就是mmcv-full，远没有想的那么简单，但是好像搞了半天，也没那么复杂，我先等等结果。应该问题不大才对。

首先到https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple/mmcv-full/ 下载一个 mmcv-full-1.7.2.tar.gz
然后解压到项目的目录下面
```python
cd /home/xxx/OmniTrack-main/mmcv-full-1.7.2

# 彻底清理
rm -rf build/ dist/ *.egg-info/
find . -name "*.so" -delete
```
设置环境变量，注意 cuda的位置可能不一样，如果一样就这么设置
```python
export MMCV_WITH_OPS=1
export CUDA_HOME=/usr/local/cuda-11.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```
设置好了之后
**手动编译（关键！）**   这个build_ext 选项很重要，缺了就不得行

```python
python setup.py build_ext --inplace

# 等待 ~~
# 出现 copying build/lib.linux-x86_64-cpython-310/mmcv/_ext.cpython-310-x86_64-linux-gnu.so -> mmcv
```
```python
ls -l mmcv/_ext*.so
# -rwxrwxr-x 1 ctt ctt 16129456  9月 24 23:23 mmcv/_ext.cpython-310-x86_64-linux-gnu.so

pip install -e . --no-deps
# Successfully uninstalled mmcv-full-1.7.2


# 验证一下
python -c "
from mmcv.ops import RoIPool, RoIAlign
import mmcv                                                     
print('mmcv-full dd！')
print('mmcv version:', mmcv.__version__)
print('RoIAlign available:', hasattr(RoIAlign, 'forward'))

```

```python
# https://github.com/state-spaces/mamba/releases
到这里选一个合适版本的mamba
pip install mamba_ssm-2.0.4+cu118torch2.0cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```
--------------------------------

环境装好了，是时候跑一跑代码，see see color
```python
# train
bash local_train.sh JRDB_OmniTrack

# test
bash local_test.sh JRDB_OmniTrack path/to/checkpoint
```
```python
# 果然有 Color,上来就给我来个 iou3d_nms_cuda 没有编译，吓我一跳，还以为是mmcv的问题。
# 结果还是一样，但是这个包藏得很深，我在CenterPoint copy 过来了应该可以用


# copying build/lib.linux-x86_64-cpython-310/iou3d_nms_cuda.cpython-310-x86_64-linux-gnu.so ->
cd /home/xxx/OmniTrack-main/projects/mmdet3d_plugin/ops/iou3d_nms
python setup.py build_ext --inplace
```

在OmniTrack模型调通之后，我们开始着手UmdaTrack的部署。

# UMDATrack
## Install the environment

Create and activate a conda environment:
```
conda create -n UMDATrack python=3.8
conda activate UMDATrack
```
Then install the required packages:
```
bash install.sh
pip install -r requirements.txt
```

## Set project paths

Run the following command to set paths for this project
```
python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir ./output
```
After running this command, you can also modify paths by editing these two files
```
lib/train/admin/local.py  # paths about training
lib/test/evaluation/local.py  # paths about testing
```

## Dataset Preparation

Put the tracking datasets in ./data. It should look like this:
```
${PROJECT_ROOT}
 -- data
     -- lasot
         |-- airplane
         |-- basketball
         |-- bear
         ...
     -- got10k
         |-- test
         |-- train
         |-- val
     -- coco
         |-- annotations
         |-- images
     -- trackingnet
         |-- TRAIN_0
         |-- TRAIN_1
         ...
         |-- TRAIN_11
         |-- TEST
     -- got10k_dark
         |-- test
         |-- train
         |-- val   
     -- got10k_haze
         |-- test
         |-- train
         |-- val 
     -- got10k_rainy
         |-- test
         |-- train
         |-- val         
``` 
The rainy haze and dark datasets are now available in [BaiduNetdisk](https://pan.baidu.com/s/1sEn0E3-Kt1X5KZYYovIYYA?pwd=es5c) and [huggingface](https://huggingface.co/datasets/WatcherBrR0/synthetic_datasets)

## Training
re-trained foundation model and initial pseudo-labels in [BaiduNetdisk](https://pan.baidu.com/s/1Xsn45GZEI35vkv6jEQ0ZHA?pwd=wi9a) 
or [Google Drive](https://drive.google.com/drive/folders/1fondgxHRdglg9JZkg_UkfqqSUmhqLUA9?usp=sharing) 
which is based on vanilla ViT-Base and put it under  `$PROJECT_ROOT$/pretrained_models`.   
Run the command below to train the model:


first step:
CUDA_VISIBLE_DEVICE=0,1,2,3 python tracking/train.py --script UMDATrack --config vit_256_ep250_all --save_dir ./outputs --mode multiple --nproc_per_node 4  --use_wandb 0
The second stage:
CUDA_VISIBLE_DEVICES=0,1,2,3 python tracking/train.py --script UMDATrack --config vit_256_ep50_dark --save_dir ./outputs --mode multiple --nproc_per_node 4  --use_wandb 0

test and statistic:
python tracking/test.py UMDATrackJSSD vit_256_ep50_dark --dataset dtb70_dark --runid 0001 --ep 40 --save_dir outputs
python tracking/analysis_results.py # need to modify tracker configs and names

In our model implement: the result is like here:
<img width="691" height="68" alt="image" src="https://github.com/user-attachments/assets/587bf277-f7e2-46c4-87b1-b4d497d71255" />
