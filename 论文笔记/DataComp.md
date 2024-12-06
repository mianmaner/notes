#### 前言

多模态数据集作为 MM-LLM 发展的关键组成部分，没有得到充分的研究关注。为了弥补这一不足，我们引入了 DataComp，这是一个围绕 Common Crawl 的新的128亿的图像文本对数据集的 benchmark

提供了一个全面的 benchmark 和数据池

#### 架构

![image-20240510210619835](https://gitee.com/mianmann/drawing-bed-warehouse/raw/master/img/image-20240510210619835.png)

#### Benchmark

**比较设计**

讲了一些约束

如何得到训练集：① 数据筛选，构建 CommonPool ② 可以使用自带的数据集

设置了四个规模构建 DataComp

**构建 CommonPool**

CommonPool 的构建有四个步骤：url 提取和数据下载、NSFW 检测、评估集去重、人脸模糊处理

**自带数据（BYOD）**

提供了一个单独的 DataComp 轨道，允许自行组合多个数据流

#### Baselines

进行了众多筛选的 Baseline 实验：
① No filtering 
② Random subsets 
③ Basic filtering
④ CLIP score and LAION filtering
⑤ Text-based filtering
⑥ Image-based filtering

![image-20240510211804272](https://gitee.com/mianmann/drawing-bed-warehouse/raw/master/img/image-20240510211804272.png)

#### 结果

![image-20240510213603981](https://gitee.com/mianmann/drawing-bed-warehouse/raw/master/img/image-20240510213603981.png)