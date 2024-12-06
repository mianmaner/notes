## 多模态串讲

### CLIP

**CLIP**的核心是文本-图像对。模型结构其实非常简单：包括两个部分，即**文本编码器（Text Encoder）**和**图像编码器（Image Encoder)**。Text Encoder选择的是**Text Transformer**模型；Image Encoder选择了两种模型，一是基于CNN的**ResNet**（对比了不同层数的ResNet），二是基于Transformer的**ViT**

![img](https://pic2.zhimg.com/v2-c2056ec1cb1b83a9b2d379366727f3a1_r.jpg)

我们可以得到一个 $n\times n$ 的相似分数矩阵，这里对角线上都是正样本，其余都为负样本。这样正负样本就可以用来训练 Text Encoder 和 Image Encoder 了

按上图所示，我们需要最大化对角线中蓝色的数值，最小化其它非对角线的数值。优化目标 ITC 可以写为：
$$min(\sum_{i=1}^N\sum_{j=1}^N(I_i\cdot T_j)_{(i\ne j)}-\sum_{i=1}^N(I_i\cdot T_j))$$

(2)和(3)也展示了CLIP如何进行 zero-shot 的图片分类任务

### BLIP

#### 总体介绍

**BLIP** 一大贡献在于将自然语言理解和自然语言生成任务融合形成了多模态通用模型（相同颜色部分是共享的，可以理解为同一个）

#### 核心思想

BLIP 预训练期间共同优化了三个目标，其中两个基于理解的目标和一个基于生成的目标。每个图像-文本对仅需要一次通过计算较重的视觉Transformer的正向传递，而三次通过文本转换器的正向传递，以激活不同的结构以计算如下所述的三个损失函数

#### 模型结构

![img](https://pic1.zhimg.com/v2-75eb42ba0027c8ebb18971737a34fb88_r.jpg)

- **图像文本对比损失 (ITC)** 激活 Image encoder 和 Text encoder。其目的是对齐视觉transformer和文本transformer的特征空间，通过鼓励正图像-文本对具有相似的表示来实现。事实证明，这是提高视力和语言理解的有效目标。

- **图像文本匹配损失（ITM）**激活 Image-grounded Text encoder。它的目的是学习图像-文本多模态表示，捕捉视觉和语言之间的细粒度对齐。ITM是一个二分类任务，给定其多模态特征，模型使用ITM头 (线性层) 来预测图像-文本对是正 (匹配) 还是负 (不匹配)。

- **语言建模损失（LM）**激活 Image-grounded Text decoder，该解码器旨在生成给定图像的文本描述。它优化了交叉熵损失，从而训练模型以自回归方式最大化文本的可能性。在计算损失时，作者采用0.1的标签平滑。与广泛用于VLP的MLM损失相比，LM使模型具有将视觉信息转换为连贯字幕的泛化能力。

#### CapFilt

研究利用了大量从网络上自动收集的图像和文本对。但是，这些文本通常无法准确描述图像的视觉内容，从而使它们成为噪声，对于学习视觉语言对齐不是最佳的

作者提出了**字幕和过滤（Captioning and Filtering，CapFilt）**，这是一种提高文本语料库质量的新方法。上图给出了CapFilt的图示。它引入了两个模块：一个用于生成给定web图像的字幕的字幕器，以及一个用于去除噪声图像-文本对的过滤器。字幕器和过滤器都是从同一个预训练过的MED模型中初始化的，并在COCO数据集上单独微调。微调是一个轻量级的过程

具体地说，字幕器是一个基于图像的文本解码器。它与LM目标相结合，对给定图像的文本进行解码。给定web图像，字幕器生成合成字幕。过滤器是一个基于图像的文本编码器。它与ITC和ITM的目标相结合，以了解文本是否与图像匹配。

### BLIP-2

#### 总体介绍

训练大尺度视觉语言预训练模型成本比较高，**BLIP-2** 通过轻量级两阶段预训练模型 Querying Transformer 缩小模态之间 gap，第一阶段从冻结图像编码器学习视觉语言表征，第二阶段基于冻结语言模型，进行视觉到语言生成学习

本文提出方法基于现有高质量视觉模型及语言大模型进行联合训练，为减少计算量及防止遗忘，作者对预训练模型进行frozen，为了将两任务对齐，作者提出Querying Transformer (Q- Former) 预训练，如图1，其将有用视觉特征传递至LLM输出目标文本

#### 模型结构

![img](https://img-blog.csdnimg.cn/be082074765a4f459a61c08a8a6b9692.png#pic_center)

Q-Former 包括两个共享 self-attention 层的 transformer 子模块：图像transformer（Q-Former 左半部分）与 frozen image encoder 相互作用提取视觉特征；文本 transformer（Q-Former 右半部分）可作为文本编码器，也可作为文本解码器

- **图像文本对比学习（ITC）**
  计算 image transformer 输出 query 表征（与可学习query长度相同）与 text transformer 输出文本表征中[CLS] token相似性，选取最大值作为图像文本对相似度，为防止信息泄露，作者使用单模态self-attention mask，query与text不能互相可见，防止从文本直接学习；由于image encoder进行frozen，显存释放，可以使用batch负样本而不用像BLIP中使用队列
- **基于图像文本生成（ITG）**
  ITG根据输入图像训练Q-Former生成文本，由于Q-Former不允许image encoder与text token直接交互，文本生成所需信息通过query进行提取，通过self-attention进行传递至text token，因此query需要捕获文本相关所有信息。作者使用多模态因果self-attention mask控制query-text交互，query无法获取text token，当前text token 可获取所有query及其之前text token。作者将【CLS】token替换为【DEC】token 作为解码任务标记
- **图文匹配（ITM）**
  ITM为了学习精细化图像文本匹配，作者使用bi-dirention self-atttention mask，所有query与text相互可见，因此输出的query embedding $Z$捕获多模态信息，$Z$通过二类线性分类器获取logit，logit均值为匹配得分，作者使用《Align before Fuse》中难例负样本挖掘策略创建负样本对

#### LLM实现视觉到语言生成

![img](https://img-blog.csdnimg.cn/4b63526d72c949aa924da3c30cbc8497.png#pic_center)

