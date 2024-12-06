提出了一个新的用来判断数据质量的指标 VAS

#### 出发点

因为高质量的数据并不总是最具信息性或代表性的。在某些情况下，某些特征可以在比其他特征少得多的样本下轻松学习，因此添加更多具有此类特征的高质量数据很难进一步改进最终模型

我们提出方差对齐评分（VAS），旨在找到训练数据中信息最丰富的子集，其图像-文本对之间的（交叉）协方差与参考分布的（交叉）协方差最一致

#### 示意图

![image-20240511171741533](https://gitee.com/mianmann/drawing-bed-warehouse/raw/master/img/image-20240511171741533.png)

VAS 筛选高信息性数据、CLIP score 筛选高质量数据

#### VAS

为了衡量整体多模态对比训练中单个样本的信息量，我们引入了一种新颖的数据分布感知指标，称为方差对齐分数（VAS）

<img src="https://gitee.com/mianmann/drawing-bed-warehouse/raw/master/img/image-20240511172045687.png" alt="image-20240511172045687" style="zoom:50%;" />

良好的嵌入模型下，方差或交叉方差项 $\frac{1}{N}\sum_i \bar f_{m_1}(x_i^{m_1})\bar f_{m_2}(x_i^{m_2})^T$ 是指示信息量的合适标准

#### 主要策略

![image-20240511172807043](https://gitee.com/mianmann/drawing-bed-warehouse/raw/master/img/image-20240511172807043.png)

我们采用两阶段方法，其中第一阶段我们删除CLIP分数非常低的样本，然后对剩余数据应用VAS以执行更细粒度的选择

我们发现使用 ImageNet1k 作为测试代理可以取得良好的性能。除了使用外部数据集 ImageNet-1k 之外，我们还提出了一种称为 VAS-D（动态）的启发式方法，该方法仅使用训练数据本身作为先验。我们称其为“动态”

<img src="https://gitee.com/mianmann/drawing-bed-warehouse/raw/master/img/image-20240511174106472.png" alt="image-20240511174106472" style="zoom:50%;" />

在实践中，我们发现视觉 CLIP 模型产生了更好的性能，这可能表明文本嵌入在恢复中不太准确。因此，我们选择使用仅视觉信息

#### 数据建立

使用 CLIP 模型作为教师模型用来恢复带有误差的 groundtruth 向量

众多理论细节暂时先不研究