### PCA

主成分分析（Principal Components Analysis，简称PCA）是最重要的数据降维方法之一。

### PCA与最大可分性

对于 $X=[x_1,x_2,\cdots,x_n]$，我们希望 $X$ 从 $n$ 维降到 $n'$ 维，同时希望信息损失最少

![PCA降维算法; PCA与最大可分性;](https://img-blog.csdnimg.cn/img_convert/e7e627e2b55e32308e51ee253072b7c4.png)

对PCA算法而言，我们希望找到小于原数据维度的若干个投影坐标方向，把数据投影在这些方向，获得压缩的信息表示

### 原理

#### 基变换

![image-20240422201831563](https://gitee.com/mianmann/drawing-bed-warehouse/raw/master/img/image-20240422201831563.png)

#### 方差

在本文的开始部分，我们提到了，降维的目的是希望压缩数据但信息损失最少，也就是说，我们希望投影后的数据尽可能分散开。在数学上，这种分散程度我们用方差来表达，方差越大，数据越分散

#### 协方差

![image-20240422203702664](https://gitee.com/mianmann/drawing-bed-warehouse/raw/master/img/image-20240422203702664.png)

#### 协方差矩阵

(1) 构造

![image-20240422203951095](https://gitee.com/mianmann/drawing-bed-warehouse/raw/master/img/image-20240422203951095.png)

(2) 协方差矩阵对角化

综上，我们的目的是要对原始数据 $X$ 做 PCA 后，得到的 $Y$ 的协方差矩阵 $D$ 的各个方向方差最大，协方差为 0

设 $X$ 的协方差矩阵为 $C$，有 $D=PCP^T$。现在我们的目标就是要找一个矩阵 $P$，满足 $PCP^T$ 是一个对角矩阵，并且对角元素按照从大到小排列，那么 $P$ 的前 $K$ 行组成的矩阵乘以 $X$ 就使得 $X$ 从 $N$ 维降到了 $K$ 维并满足上述优化条件

### 算法

![PCA降维算法; PCA的算法步骤;](https://img-blog.csdnimg.cn/img_convert/c6a0b8f2e9a35fb07a40cd82879b3075.png)