# 李宏毅GNN入门

## GNN可以做什么

图分类、图生成（制作新药）、特殊节点分类、图表征学习、链接预测

## spatial domian

### 概念

Aggregate和Readout
![](https://gitee.com/mianmann/drawing-bed-warehouse/raw/master/img/image-20240122171609352.png)

### 模型

#### NN4G

①Aggregate：
![image-20240122171834150](https://gitee.com/mianmann/drawing-bed-warehouse/raw/master/img/image-20240122171834150.png)

②Readout：
![image-20240122174515729](https://gitee.com/mianmann/drawing-bed-warehouse/raw/master/img/image-20240122174515729.png)

#### DCNN

①Aggregate：
![image-20240122174700056](https://gitee.com/mianmann/drawing-bed-warehouse/raw/master/img/image-20240122174700056.png)

②Readout：
![image-20240122174929217](https://gitee.com/mianmann/drawing-bed-warehouse/raw/master/img/image-20240122174929217.png)

#### MoNet

![image-20240122175038524](https://gitee.com/mianmann/drawing-bed-warehouse/raw/master/img/image-20240122175038524.png)

#### GraghSAGE

![image-20240122175011212](https://gitee.com/mianmann/drawing-bed-warehouse/raw/master/img/image-20240122175011212.png)

#### GAT

#### ![image-20240122175122672](https://gitee.com/mianmann/drawing-bed-warehouse/raw/master/img/image-20240122175122672.png)GIN

提出了一个合理的Aggreation公式，论证了mean和pool都是不合理的

![image-20240122175146695](https://gitee.com/mianmann/drawing-bed-warehouse/raw/master/img/image-20240122175146695.png)

## spactral domain

> 前提条件：无向图
> 详见https://blog.csdn.net/yyl424525/article/details/100058264

### 概念

(1)拉普拉斯矩阵：主要应用在图论中，作为一个图的矩阵表示
![img](https://note.youdao.com/yws/api/personal/file/WEB71b23675eda61d5a0f238006ec8ef43d?method=download&shareKey=34d5b29456242159ea1875260bafcfab)

(2)常用的几种拉普拉斯矩阵

①普通形式的拉普拉斯矩阵

$L=D-A$：矩阵元素表示如下，$diag(v_i)$表示定点$i$的度
![image-20240123172721927](https://gitee.com/mianmann/drawing-bed-warehouse/raw/master/img/image-20240123172721927.png)

②对称归一化的拉普拉斯矩阵

$L^{sys}=D^{-1/2}LD^{-1/2}=I-D^{1/2}AD^{-1/2}$：矩阵元素定义如下
![image-20240123173421158](https://gitee.com/mianmann/drawing-bed-warehouse/raw/master/img/image-20240123173421158.png)

③随机游走归一化的拉普拉斯矩阵

$L^{rw}=D^{-1}L=I-D^{-1}A$

![image-20240206202745892](https://gitee.com/mianmann/drawing-bed-warehouse/raw/master/img/image-20240206202745892.png)

> **无向图的拉普拉斯矩阵性质：**
> ①拉普拉斯矩阵是半正定矩阵（最小特征值大于等于0）
> ②特征值中0出现的次数就是图连通区域的个数
> ③最小特征值是0，因为拉普拉斯矩阵（普通形式：L = D − A L=D-AL=D−A）每一行的和均为0，并且最小特征值对应的特征向量是每个值全为1的向量
> ④最小非零特征值是图的代数连通度

> 特征值与特征向量：如果向量v与变换A满足$Av=\lambda v$，则称向量v是变换A的一个特征向量，λ是相应的特征值

(3)特征分解（谱分解）：是将矩阵分解为由其特征值和特征向量表示的矩阵之积的方法（只有对可对角化矩阵或有n个线性无关的特征向量的矩阵才可以施以特征分解）

(4)时域和频域

- ①时域：真实量到的信号的时间轴，代表真实世界
  ②频域：为了做信号分析用的一种数学手段
- 频域分解示意图
  <img src="https://gitee.com/mianmann/drawing-bed-warehouse/raw/master/img/image-20240218122712393.png" alt="image-20240218122712393" style="zoom:50%;" />

(5)傅里叶变换

- 回顾：任意一个周期函数可以由若干个正交函数（sin和cos）的线性组合构成，又由欧拉公式，可以得到$f(x)=\sum_{n=-\infty}^\infty c_n\cdot e^{i\frac{2\pi nx}{T}}$。又因为非周期函数可以用某个周期函数$f_T(x)$当$T\rightarrow\infty$时表示，又有
  ![image-20240208234836437](https://gitee.com/mianmann/drawing-bed-warehouse/raw/master/img/image-20240208234836437.png)
  所以$F(\omega)$就是$f(x)$的连续形势的傅里叶变换
  
- 图上的傅里叶变换：
  ![img](https://note.youdao.com/yws/api/personal/file/WEB2a446d869c2aec5d7e49ed3c5880d9e9?method=download&shareKey=50a455d7ef31604a5d12148912deaf3b)
  
- 定义图上的傅里叶变换$F(\lambda_1)=\sum_{i=1}^Nf(i)u_1(i)$，推广到矩阵形式为$\hat f=U^Tf$如下：

  ![image-20240219110343118](https://gitee.com/mianmann/drawing-bed-warehouse/raw/master/img/image-20240219110343118.png)

  ①$f(i)$即为，$u_l(i)$即表示第$l$个特征向量的第$i$个分量
  ②$\lambda$就对应于传统意义上的$\omega$，$\hat f$表示$f$的傅里叶变换
  ③$f$的图傅里叶变换就是与$\lambda_l$对应的特征向量$u_l$进行内积计算
  
- 图的傅里叶逆变换

  传统的傅里叶逆变换即为对$\omega$求积分，迁移到图上即为对特征值$\lambda_l$求和为$f(i)=\sum_{l=1}^N\hat f(\lambda_l)u_l(i)$，即为$f=U\hat f$如下：
  ![image-20240219112236070](https://gitee.com/mianmann/drawing-bed-warehouse/raw/master/img/image-20240219112236070.png)

(6)卷积定理

- 卷积定理：函数卷积的傅里叶变换就是函数傅里叶变换的卷积，即对于函数f和g二者的卷积是其函数傅里叶变换乘积的逆变换

  ![image-20240219112551117](https://gitee.com/mianmann/drawing-bed-warehouse/raw/master/img/image-20240219112551117.png)
  所以，对于图上f和卷积核g的卷积可以表示为$(f*g)_G=U((U^Tg)\cdot(U^Tf))$，又因为傅里叶变换的结果$U^Tg$和$U^Tf$都是一个列向量，所以很多论文中的图卷积公式也写作$(f*g)_G=U((U^Tg)\odot (U^Tf))$，$\odot$表示哈达玛积，对于两个向量就是进行内积计算，对于维度相同的两个矩阵，就是说对应元素的乘积运算

- 如果把$U^Tg$整体看作可学习的卷积核，这里我们把它写作$g_\theta$，最终图上的卷积公式即为$(f*g)_G=Ug_\theta U^Tf$

> **为什么拉普拉斯矩阵的特征向量可以作为傅里叶变换的基？特征值表示频率？**
> 图上进行傅里叶变换时，拉普拉斯矩阵是对称矩阵，所以有n个线性无关的特征向量（由线性代数的知识可以知道n维空间中n个线性无关的向量可以构成空间的一组基，而且拉普拉斯矩阵的特征向量还是一组正交基）。这与传统意义上的若干个正交函数理念一致，因此可以构成傅里叶变换的一组基，而其对应的特征值就是傅里叶变换的频率。

###   模型

#### Spectral CNN

> J. Bruna, W. Zaremba, A. Szlam, and Y. LeCun, “Spectral networks and locally connected networks on graphs,” in Proceedings of International Conference on Learning Representations, 2014（首次提出）

简单的将$g_\theta$看作一个可学习参数的集合$g_\theta =\Theta_{i,j}^k$，图卷积层定义为：

![image-20240219141635046](https://gitee.com/mianmann/drawing-bed-warehouse/raw/master/img/image-20240219141635046.png)

弊端：计算复杂、非局部性链接、卷积核需要N个参数，当节点N很大时是不可取的

#### ChebNet

> M. Defferrard, X. Bresson, and P. Vandergheynst, “Convolutional neural networks on graphs with fast localized spectral filtering,”in Advances in Neural Information Processing Systems, 2016

利用Chebyshev多项式拟合卷积核的方法，来降低计算复杂度

#### 1stChebNet-GCN

> T. N. Kipf and M.Welling, “Semi-supervised classification with graph convolutional networks,” in Proceedings of the International Conference on Learning Representations, 2017（GCN的开山之作）
