### 发表

2021-04-19 WWW '21: The Web Conference 2021

### 任务

异常检测->根因定位（服务实例级别）（针对延迟问题）

### 贡献

- MicroRank 是第一种通过正常和异常trace提取信息来对微服务应用程序进行根本原因定位的方法
- 我们提出了一种基于扩展频谱分析的微服务环境中新颖的根本原因定位方法
- 我们将 OpenTelemetry trace API 纳入 Google Cloud 提供的 HipsterShop 微服务的benchmark中。我们的修改使该benchmark具有以前没有的端到端tracing能力
- 我们设计并实现了一个原型，即 MicroRank，来定位微服务系统中延迟问题的根本原因。我们基于一个广泛使用的开源微服务系统和一个总共有157个故障的生产微服务系统进行了大量的实验。实验结果证明了 MicroRank 相对于最先进方法的有效性

### 概述

提出MicroRank 的新方法，该方法基于扩展频谱分析来识别微服务系统中的延迟问题。它主要包括四个程序，包括：

- Anomaly Detector（异常检测模块）
- Data Preparator（数据准备模块)
- PageRank Scorer （PageRank得分计算模块）
- Weighted Spectrum Rank（加权频谱排名模块）

一旦 MicroRank 中的Anomaly Detector检测到延迟问题，就会触发原因定位程序。 Data Preparator 首先区分哪些trace是异常的，并注意服务实例和trace的关系。然后，MicroRank 的 PageRank Scorer 模块使用异常和正常trace信息作为输入，并区分不同trace对扩展频谱技术的重要性。最后，Weighted Spectrum Rank根据来自 PageRank Scorer的加权频谱信息输出潜在根本原因的排名列表

### 从Trace中提取信息

- Latency and Handling Time。trace数据包括每条trace的端到端延迟和一条trace中每个操作的处理时间
- Trace Coverage Graph。由于一条trace中的每个span都包含父子关系信息和服务实例信息，从而可以构建出这样一个树，完整的体现了所有trace的父子调用关系

![image-20231109141856259](https://gitee.com/mianmann/drawing-bed-warehouse/raw/master/img/image-20231109141856259.png)

- 调用图。根据Trace Coverage Graph中所有节点的服务实例，我们就可以获得不同服务操作之间的调用图

基于以上信息，微服务系统中的延迟问题定位问题的形式化描述如下。给定时间窗口中的一组trace，即 $T=(T_1,\dots,T_n) $，其中 $T_i = (O_{i1},\cdots, O_{im})$（ Oij 是Ti 中的一个Operation）。我们得到这些trace的端到端延迟，即 $ L = (L_1,\cdots, L_n ) $和覆盖图 G = (V , E)，其中 V 是服务实例操作的集合，即 Oij ∈ V。E是调用关系的集合。找出正常trace: $T_n=(T_{n1} ,\cdots,T_{nk} )$和异常trace $Ta=(T_{a1},\cdots,T_{an-k})$。此外，根据Tn、Ta和G，找到与Oij 相关的根本原因服务实例。最后将与根本原因直接相关的服务实例的排名高于与根本原因无关的服务实例的排名

### 基于频谱的故障定位

给出元组 (Oef , Oep , Onf , Onp ) 。其中，Oef 表示覆盖程序元素 O 的失败测试用例的数量，Oep 表示覆盖程序元素 O 的通过测试用例的数量。Onf 表示未覆盖程序元素 O 的失败测试用例的数量，Onp 表示未覆盖程序元素 O 的失败测试用例的数量。根据上述元组，可以得出几个风险评估公式，例如Tarantula

### MicroRank

![image-20231109141913490](https://gitee.com/mianmann/drawing-bed-warehouse/raw/master/img/image-20231109141913490.png)

#### Anomaly Detector

提出了一种无监督且轻量级的跟踪级异常检测方法，该方法利用操作处理时间与跟踪覆盖的操作数量之间的关系。第一异常检测器根据离线正常trace数据的一段时间（例如，一小时）计算每个操作的平均处理时间 $\mu_O$及其标准差 $\sigma_O$

当监控一条Trace时，我们只需要获取一条Trace覆盖了哪些操作以及这些操作被覆盖了多少次，然后根据下面的公式计算 $L_{expected}$期待延迟：

![image-20231109141932012](https://gitee.com/mianmann/drawing-bed-warehouse/raw/master/img/image-20231109141932012.png)

其中 counto 表示该跟踪覆盖操作 o 的次数， n用于调整上界值，本文设置n=1.5。

如果期待延迟L excepted 小于该trace的真实延迟，则该trace将被确定为异常trace。一旦Anomaly Detector检测到一条异常trace，它就会触发根因定位阶段。为了避免多次检测到相同的异常状态，这里会在在每次触发后刷新检测窗口（本文中为 5 分钟）。

#### Data Preparator

对于异常trace列表或正常trace列表中的每个trace ID ，Data Preparator 通过traceID 获取其trace的提取信息。在基于此构建完整的anomalous operation-trace graph

![image-20231109141946660](https://gitee.com/mianmann/drawing-bed-warehouse/raw/master/img/image-20231109141946660.png)

#### PageRank Scorer

PageRank Scorer由正常评分器和异常评分器组成，并行处理异常和正常线索

基本思想是根据operation-trace graph（例如图 6）评估覆盖每个operation的trace的重要性，从而为每个operation赋予权重。

PageRank Scorer 的key Insights：

- 如果某个操作被更多异常trace覆盖，则它也更有可能是根本原因
- 如果异常trace覆盖较少操作，则该痕迹应被视为更重要，因为它的范围较小
- 如果某类trace出现次数较少，则应多考虑该种trace，防止从多次出现的trace中分流

这里选择Personalized PageRank，因为它是一种分析异构节点图（本文中的operation和trace）的方法。给定一个有向图 G = ⟨V ，E⟩ 包含 n 个节点和 m 个边，并且 E 包含一条有向边 ⟨s, t⟩（如果节点 s 到节点 t）。设A为n×n元素的转移矩阵，所有的Ast将组合成完整的A（Ast被定义为从s开始的随机游走在t终止的概率，反映了t相对于s的重要性）。Ast 的值可由下式计算：

![image-20231109142003310](https://gitee.com/mianmann/drawing-bed-warehouse/raw/master/img/image-20231109142003310.png)

其中 O (s) 表示 s 的出邻居（从s出发可以到达的相邻节点）

然后对于给定的偏好向量 u，Personalized PageRank 方程可以写为：

![image-20231109142016836](https://gitee.com/mianmann/drawing-bed-warehouse/raw/master/img/image-20231109142016836.png)

其中 d 是阻尼因子 (0 ≤ d < 1)， v 是偏好向量 u 的Personalized PageRank 向量 (PPV)。结果向量v表示所有节点的排名分数。

> PageRank中的相关概念：
> ①偏好向量 (Preference Vector) u：这是一个向量，它用来表示用户对网络中各个节点（或页面）的偏好程度。每个元素表示用户对特定节点的偏好权重，这些权重可以根据用户的需求来设置。比如当 u是均匀分布向量时，也就是所有元素都相等，意味着用户对所有页面都没有特别的偏好。
> ②PageRank 向量 v：这是一个向量，它代表了每个节点在 PageRank 算法中的重要性分数。
> ③Damping Factor (阻尼因子) d：这是一个介于 0 到 1 之间的数值，用于控制 PageRank 算法的随机跳跃概率。

#### Weighted Spectrum Ranker

操作O的频谱信息可以计算为：

![image-20231109142036871](https://gitee.com/mianmann/drawing-bed-warehouse/raw/master/img/image-20231109142036871.png)

其中F和P表示操作O的异常和正常PageRank得分，Nef和Nep表示操作O覆盖的异常和正常trace的数量，Nf和Np表示当前滑动窗口中异常和正常痕迹的总数

然后，MicroRank 应用频谱排名公式来计算每个操作的可疑度分数。操作名称（例如 front-1/Recv.）包含有关其所属服务实例的信息。因此，MicroRank可以为应用运营商给出一个服务实例的排名列表。表1的右半部分显示了图3中每个服务实例的加权频谱信息和更新的Tarantula分数

![image-20231109142126415](https://gitee.com/mianmann/drawing-bed-warehouse/raw/master/img/image-20231109142126415.png)

### 数据集

使用了三个数据集，分别称作A、B和C。A和B基于广泛使用的开源微服务系统Hipster-Shop。 A和B之间的主要区别在于，我们在A中每次注入一个故障，而在B中每次注入两个故障。C基于中国移动的生产微服务系统，每次注入一个故障（数据集C是2020年AIOps挑战赛第14期发布的，C基于中国移动浙江移动的真实生产微服务系统。）

![image-20231109142321772](https://gitee.com/mianmann/drawing-bed-warehouse/raw/master/img/image-20231109142321772.png)

### 实验结果

为了比较各频谱公式的有效性差异，这里总共选择了八个公式

![image-20231109142245842](https://gitee.com/mianmann/drawing-bed-warehouse/raw/master/img/image-20231109142245842.png)

列 PR 代表 MicroRank 中的 PageRank Scorer，列 SP 代表传统的基于频谱的技术，列 MR 代表 MicroRank（PageRank Scorer + 频谱分析）

> ①Top-k (R@k) 的Recall是指在所有候选服务中可以在前 k 个服务实例中定位根本原因的概率
>
> ②EXAM 分数是指 找到所有根因所需要的由操作员手动排除的误报的平均数。如果根本原因不在 Top-5 之列，就为其设置默认误报数 10。对于 ES，越小越好。

![image-20231109142156014](https://gitee.com/mianmann/drawing-bed-warehouse/raw/master/img/image-20231109142156014.png)

发现 MicroRank 受不同公式的影响较小。从结果来看，Ochiai（灰色标记）是我实验中最有效的公式。所以最终采用Ochiai作为默认频谱公式