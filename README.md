# ManiTrack-MT
This is the code of our work: ManiTrack-MT: Multi-Teacher Manifold Distillation for Robust Visual Tracking.
We will open our project step by step.


# Datasets Collection

兵马未动，粮草先行，首先进行数据集的收集，以下是所有的数据的来源
感谢各位大佬在百度网盘中保存的完好的数据，感谢！

1、 trackingnet：https://zhuanlan.zhihu.com/p/673825440
在下载完后，还需要将这些大的资源包进行组合和解压，是一个大工程，光下载就得两三天。
对于下载好的文件进行批处理，用简单的命令行就可以进行：
```python
for d in TRAIN_{0..10}; do echo "处理 $d"; (cd "$d" && 7z x -y *.zip.001 -o"../extracted_$d"); done
```
在把它们都解压完之后，发现每个extracted_TRAIN_x 中有两个文件，anno和zips，这个zips里面全是zip,因此，给定bash程序 unzipall.bash
```bash
#!/bin/bash

# 循环处理 extracted_TRAIN_0 到 extracted_TRAIN_10
for i in {0..10}; do
    dir="extracted_TRAIN_$i"
    zips_dir="$dir/zips"
    output_dir="$dir/zipdata"

    # 检查 zips 目录是否存在
    if [ ! -d "$zips_dir" ]; then
        echo "警告: $zips_dir 不存在，跳过..."
        continue
    fi

    # 创建 zipdata 目录（如果不存在）
    mkdir -p "$output_dir"

    # 遍历 zips 目录下的所有 .zip 文件
    for zip_file in "$zips_dir"/*.zip; do
        # 检查是否真的有 zip 文件（避免空匹配）
        [ -f "$zip_file" ] || continue

        # 获取 zip 文件名（不含路径）
        filename=$(basename "$zip_file")
        # 去掉 .zip 后缀，作为解压目录名
        dirname="${filename%.zip}"

        # 创建目标子目录
        target_dir="$output_dir/$dirname"
        mkdir -p "$target_dir"

        # 解压到目标子目录
        unzip -q "$zip_file" -d "$target_dir"

        echo "已解压: $zip_file -> $target_dir"
    done

    echo "✅ $dir 处理完成"
done
echo "🎉 所有解压任务完成！"
```
然后 给这个文件赋予权限
```
chmod +x unzipall.bash
bash unzipall.bash
```
等它执行完就ok

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
