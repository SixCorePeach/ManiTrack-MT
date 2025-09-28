# TrackerTT

兵马未动，粮草先行，首先进行数据集的收集，以下是所有的数据的来源
感谢各位大佬在百度网盘中保存的完好的数据，感谢！

1、 trackingnet：https://zhuanlan.zhihu.com/p/673825440
在下载完后，还需要将这些大的资源包进行组合和解压，是一个大工程，光下载就得两三天。
对于下载好的文件进行批处理，用简单的命令行就可以进行：
```python
for d in TRAIN_{0..10}; do echo "处理 $d"; (cd "$d" && 7z x -y *.zip.001 -o"../extracted_$d"); done
```

2、 UAVDark 70：   https://pan.baidu.com/s/1PTFwNoSxwZBmUSzDD3ti2A    提取码：1234
这个比较简单，也比较小 只有7.2G
这里附上 UAVDark 135：https://pan.baidu.com/s/1JcV_wTUSt9F8iBXiLCZQdQ 提取码：axci

3、lasot

4、got10k : https://pan.baidu.com/s/15iXqOEBj99S8-VTpmsLiOg   如果这个不奏效，还请访问got10k的官网寻求帮助

5、coco : 这个比较经典，所以我就没有下载，想来也没有影响，其他的数据集这么多。

6、trackingnet：这个数据集有1.14T哦，需要准备大一点的盘，另外，解压的话，使用高性能的计算机会更好一点。
trackingnet的下载地址在上方，这里不再缀诉。

7、got10k_dark：这个黑夜的是UMDATrack通过使用风格渲染的方式对got10k数据集进行数据增强，得到的结果，可以下载下来。

8、got10k_haze：同上

9、got10l_rainy：同上

下载地址在这里：https://pan.baidu.com/s/1Xsn45GZEI35vkv6jEQ0ZHA?pwd=wi9a

10、JRDB2019： you need submit the register to JDBR by https://jrdb.erc.monash.edu/dataset/

在注册完成，获取下载链接，得到数据集之后，只需要按照步骤处理即可，这里也放上处理的代码，在tools/jrdb_generation.py，

到这里，数据集已经全部处理完成，开始跑实验

2025-09-24 终于开始跑实验了，首先进行Omini的baseline模型 的train 和 test安排（使用的市JRDB和他们自己发布的）
噢对了，放上该工作的链接：https://github.com/xifen523/OmniTrack

先准备数据噢，上面已经下好了，得放在这个位置：
```python  
.../OmniTrack/data/JRDB2019    # 要在data下面建立一个 JRDB2019

cd data
mkdir JRDB2019

JRDB2019
├── test_dataset_without_labels
│   ├── calibration
│   ├── detections
│   ├── images
│   ├── pointclouds
│   └── timestamps
│   ...
├── train_dataset_with_activity
│   ├── calibration
│   ├── detections
│   ├── images
│   ├── labels
│   ├── pointclouds
│   └── timestamps
│   ...
```

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



Acknowledge

首先致谢本文的基线模型，也是本工作影响最大的模型：UMDATrack 和 OmniTrack
```
@inproceedings{yao2025umdatrack,
  title={UMDATrack: Unified Multi-Domain Adaptive Tracking Under Adverse Weather Conditions},
  author={Yao, Siyuan and Zhu, Rui and Wang, Ziqi and Ren, Wenqi and Yan, Yanyang and Cao, Xiaochun},
  booktitle={ICCV},
  year={2025}
}

@inproceedings{luo2025omniTrack,
  title={Omnidirectional Multi-Object Tracking},
  author={Kai Luo, Hao Shi, Sheng Wu, Fei Teng, Mengfei Duan, Chang Huang, Yuhang Wang, Kaiwei Wang, Kailun Yang},
  booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2025}
}
```
Thanks
https://github.com/tianweiy/CenterPoint/tree/d3a248fa56db2601860d576d5934d00fee9916eb
