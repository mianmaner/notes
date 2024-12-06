## Numpy

```python

np.eye(N,M=None) # 返回一个二维数组(N,M)，对角线的地方为1，其余的地方为0，可用于构造one-hot
np.unique(A) # 对于一维数组或者列表,unique函数去除其中重复的元素，并按元素从大到小返回一个新的元组或列表
np.triu() # 上三角形
np.tril() # 下三角形

np.random.choice()
```



## PyTorch

### 常见基本函数

```python
#
torch.ones()
torch.zeros()
torch.tensor()
torch.from_numpy()

torch.rand(*size) # 返回一个张量，包含了从[0,1)的均匀分布中抽取的一组随机数
torch.randint(low=0,high,*size) # 返回一个张量，包含了从[low,high)的均匀分布中抽取的一组随机数
torch.randn(*size) # 返回一个张量，包含了从标准正态分布中抽取的一组随机数
torch.randperm(n) # 返回0到n-1之间所有数字的一个随机排列

torch.cos()
torch.sin()
torch.tan()
torch.sqrt()
torch.var() # 返回输入张量所有元素的方差
torch.std() # 返回输入张量所有元素的标准差
torch.cat(tensors,dim=0) # 将多个张量按指定维度拼接在一起
torch.range(start,end,step) # 结果包含end
torch.arange(start,end,step) # 结果不包含end

torch.ge(input,other) # 两个相同形状的张量逐元素比较，大于等于处则赋值为True，否则为False
torch.gt() # 大于
torch.le() # 小于等于
torch.lt() # 小于
torch.eq() # 等于

tensor.numpy()
tensor.storage() # 存储信息
tensor.stride(dim) # 张量在指定维度中从一个元素跳到下一个元素中的步长 
tensor.to(*args)
tensor.squeeze() # 去掉所有维数为1的维度
tensor.unsqueeze(dim) # 在某一维度后增加一维(若为-1则自动分配剩下的)
tensor.view(dims) # 调整张量的形状(若为-1则自动分配剩下的)
tensor.scatter_(dim,index,src) # 详见下面
tensor.transpose(dim0,dim1) # 将张量的两个维度互换
tensor.permute(dims) # 将张量按期望的维度顺序排列 
tensor.expand(*sizes) # 扩展成size（因为是复制扩展，所以仅适用于单数维度tensor）
tensor.expand_as(b) # 按照b的size扩展（因为是复制扩展，所以仅适用于单数维度tensor）
tensor.repeat(*copies) # 第i个参数表示在轴i上复制i次（整体复制，适用于所有tensor）
tensor.type_as(b) # 将tensor的类型转换成b的类型


torch.utils.data.Dataloader(dataset,batch_size,shuffle=False) # dataloader的参数顾名思义，shuffle为每个epoch的数据是否打乱

```

### PyTorch中的一些Tricks

- ``tensor.scatter_()``：

- ``nn.Embedding(num_embeddings,embedding_dim)``：(num_embeddings是词典长度，embedding_dim是词向量维度)

  ```python
  embedding = nn.Embedding(10,3)
  embedding.weight
  '''
  Parameter containing:                   
  tensor([[ 1.2402, -1.0914, -0.5382],
          [-1.1031, -1.2430, -0.2571],
          [ 1.6682, -0.8926,  1.4263],
          [ 0.8971,  1.4592,  0.6712],
          [-1.1625, -0.1598,  0.4034],
          [-0.2902, -0.0323, -2.2259],
          [ 0.8332, -0.2452, -1.1508],
          [ 0.3786,  1.7752, -0.0591],
          [-1.8527, -2.5141, -0.4990],
          [-0.6188,  0.5902, -0.0860]], requires_grad=True)
  '''
  ```

  使用：`embedding(input)`是去embedding.weight中取对应index的词向量，且input必须是LongTensor类型

  ```python
  # an Embedding module containing 10 tensors of size 3
  embedding = nn.Embedding(10, 3)
  # a batch of 2 samples of 4 indices each
  input = torch.LongTensor([[1,2,4,5],[4,3,2,9]])
  embedding(input)
  '''
  tensor([[[-0.0251, -1.6902,  0.7172],
           [-0.6431,  0.0748,  0.6969],
           [ 1.4970,  1.3448, -0.9685],
           [-0.3677, -2.7265, -0.1685]],
  
          [[ 1.4970,  1.3448, -0.9685],
           [ 0.4362, -0.4004,  0.9400],
           [-0.6431,  0.0748,  0.6969],
           [ 0.9124, -2.3616,  1.1151]]])
  '''
  ```

- 激活函数和损失函数的组合：
  ①`nn.LogSoftmax`和`nn.NLLLoss`的组合等同于使用`nn.CrossEntropyLoss()`
  ②`nn.Sigmoid`和`nn.BCELoss`的组合等同于使用`nn.BCEWithLogitsLoss()`

- `torch.Conv2d(channels, output, (height,width))`

  ```python
  # 具体用法详见TextCNN实战（nn.Modulelist管理不同的卷积层Conv2d）
  import torch
  
  x = torch.randn(3,1,5,4)
  print(x)
  
  conv = torch.nn.Conv2d(1,4,(2,3))
  res = conv(x)
  
  print(res.shape)    # torch.Size([3, 4, 4, 2])
  
  '''
  输入：x[ batch_size, channels, height_1, width_1 ]
  batch_size，一个batch中样本的个数 3
  channels，通道数，也就是当前层的深度 1
  height_1， 图片的高 5
  width_1， 图片的宽 4
  
  卷积操作：Conv2d[ channels, output, height_2, width_2 ]
  channels，通道数，和上面保持一致，也就是当前层的深度 1
  output ，输出的深度 4【需要4个filter】
  height_2，卷积核的高 2
  width_2，卷积核的宽 3
  
  输出：res[ batch_size,output, height_3, width_3 ]
  batch_size,，一个batch中样例的个数，同上 3
  output， 输出的深度 4
  height_3， 卷积结果的高度 4
  width_3，卷积结果的宽度 2
  '''
  ```

- 关于训练：

  ```python
  model = Word2Vec()
  
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=0.001)
  
  # Traning
  for epoch in range(3000):
      input_batch, target_batch = random_batch() 
      input_batch = torch.Tensor(input_batch)
      target_batch = torch.LongTensor(target_batch)
  
      optimizer.zero_grad() # 优化器梯度清零（因为不清零的话后续进行loss.backward梯度会叠加）
      output = model(input_batch) 
  
      loss = criterion(output, target_batch)
      if (epoch + 1) % 1000 == 0:
      	print(f'Epoch: {epoch + 1}, cost: {loss}')
  
      loss.backward() # 反向传播，计算梯度
      optimizer.step() # 进行一步梯度下降
  ```

- 自定义Dataset用于DataLoader

  ```python
  # torch.utils.data.Dataset
  class CustomDataset(data.Dataset):
      def __init__(self):
          # TODO
          # 1. Initialize file path or list of file names.
          pass
      
      def __getitem__(self, index):
          # TODO
          # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
          # 2. Preprocess the data (e.g. torchvision.Transform).
          # 3. Return a data pair (e.g. image and label).
          pass
      
      def __len__(self):
          # You should change 0 to the total size of your dataset.
          return 0
  ```

- `nn.Module`中的`self.register_buffer`：在内存中定义一个常量，模型保存和加载时可以写入和读出；但模型训练时不会更新该常量的值，只可人为的修改

  ```python
  class PositionalEncoding(nn.Module):
      """Transformer中的位置编码"""
      def __init__(self, d_model, dropout=0.1, max_len=5000):
          super(PositionalEncoding, self).__init__()
          self.dropout = nn.Dropout(p=dropout)
          pe = torch.zeros(max_len, d_model)
          # 因为div_term是序列，所以pos矩阵需要添加一个维度来与其匹配相乘
          position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
          # 下面的过程即为原论文中的pe计算公式
          div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)) / d_model)
  
          # 原论文中计算公式2i+1和2i均对应的2i，所以这里均为div_term
          # 偶数维度
          pe[:, 0::2] = torch.sin(position * div_term)
          # 奇数维度
          pe[:, 1::2] = torch.cos(position * div_term)
          
          pe = pe.unsqueeze(0).transpose(0,1)
          self.register_buffer('pe', pe)
  
      def forward(self, x):
          x = x + self.pe[:x.size(0), :]
          return self.dropout(x)
  ```

- `contiguous()`用法：如果在view之前使用了transpose、permute等，需要用contiguous返回一个contiguous copy（因为有些tensor并不是占用一整块内存，而是由不同的数据块组成，而tensor的view()操作依赖于内存是整块的，这时只需要执行contiguous()这个函数，把tensor变成在内存中连续分布的形式）

  ```python
  context = context.transpose(1,2).contiguous().view(batch_size, -1, n_heads * d_v)
  ```

- `nn.Embedding`要求的输入是`LongTensor`，所以对于输入的原始数据应该是`LongTensor`的形式，但经过`Embedding`之后就是`FloatTensor`了，所以模型的输出时`FloatTensor`，又因为在计算`Loss`要求类型一致，所以我们的`labels`一般处理成`FloatTensor`

- PyTorch中`CrossEntropyLoss`和`NLLLoss`所期望的target都是类别值，而不是One-hot编码格式（这个使用One-hot是否会有影响我并没有去验证过，目前没有遇到过因为这里而影响结果的）

- 因此 `Batch Normalization` 层恰恰插入在 Conv 层或全连接层之后，而在 ReLU等激活层之前。而对于 dropout 则应当置于激活层之后
