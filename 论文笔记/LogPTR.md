### 挑战

① 现有的日志解析器往往忽略变量的类别
② 现有日志解析器泛化能力差

提出基于传统深度学习模型的日志解析器 LogPTR

### 框架

![image-20240908232128588](https://gitee.com/mianmann/drawing-bed-warehouse/raw/master/img/image-20240908232128588.png)

先用 Wordpiece 将日志消息标记为子词序列，然后输入到 Embedding 层，然后用 Bi-LSTM 层去捕捉单词之间的依赖关系

指针机制：通过创建指向输入序列中单词的指针来生成变量感知日志模板（论文中的解释不太清晰，感觉就是在说 RNN 的前向传播）

### 总结

这篇是基于 seq2seq 模型的日志解析器，属于传统深度学习模型类型的，能达到 SOTA 效果