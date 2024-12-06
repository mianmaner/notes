> 详情见：https://zhuanlan.zhihu.com/p/641755465

#### 架构：

![image-20240310111742025](https://gitee.com/mianmann/drawing-bed-warehouse/raw/master/img/image-20240310111742025.png)

keyword LLM（关键词转化模型）：用于将客户口语化的诉求转化为法律行业关键词

law LLM（向量匹配模型）：BERT Embedding模型，用于直接根据口语化表达检索相关法律知识

chatlaw LLM（融合大模型）：多检索出的结果进行分析，提取关键内容，过滤不关键内容，生成相关回复

#### 要点：

训练数据构造仔细、重要

向量检索+关键词检索相结合

提出一种self-attention的方法

#### 结论：

包括GPT-4在内的通用大模型效果比较差，Elo机制下，本文提出的方法在表现上超过了GPT-4

#### 模型详细：

![img](https://pic4.zhimg.com/v2-bab545097a3656b8749d2d4481c58643_r.jpg)