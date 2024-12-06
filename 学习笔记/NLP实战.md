## Word2Vec

### 实战代码

```python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


def random_batch():
    random_inputs = []
    random_labels = []
    random_index = np.random.choice(
        range(len(skip_grams)), batch_size, replace=False)

    for i in random_index:
        # random_inputs are one-hot encoding
        random_inputs.append(np.eye(voc_size)[skip_grams[i][0]])
        random_labels.append(skip_grams[i][1])

    return random_inputs, random_labels


# Model
class Word2Vec(nn.Module):
    def __init__(self):
        super(Word2Vec, self).__init__()
        self.W = nn.Linear(voc_size, embedding_size, bias=False)
        self.WT = nn.Linear(embedding_size, voc_size, bias=False)

    def forward(self, X):
        # X: [batch_size,voc_size]
        hidden_layer = self.W(X)
        output_layer = self.WT(hidden_layer)
        return output_layer


if __name__ == '__main__':
    batch_size = 2
    embedding_size = 2

    sentences = ["apple orange fruit", "apple banana fruit", "banana orange fruit",
                 "dog cat animal", "cat monkey animal", "monkey dog animal"]

    word_sequence = " ".join(sentences).split()
    word_list = list(set(word_sequence))
    word_dict = {w: i for i, w in enumerate(word_list)}
    voc_size = len(word_list)

    # make skip-gram of one size window
    skip_grams = []
    for i in range(1, len(word_sequence)-1):
        target = word_dict[word_sequence[i]]
        context = [word_dict[word_sequence[i-1]], word_dict[word_sequence[i+1]]]
        for w in context:
            skip_grams.append([target, w])

    model = Word2Vec()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Traning
    for epoch in range(3000):
        input_batch, target_batch = random_batch() # input_batch: [batch_size, voc_size]
        input_batch = torch.Tensor(input_batch)
        target_batch = torch.LongTensor(target_batch)

        optimizer.zero_grad()
        output = model(input_batch) # output: [batch_size, voc_size]

        loss = criterion(output, target_batch)
        if (epoch + 1) % 1000 == 0:
            print(f'Epoch: {epoch + 1}, cost: {loss}')

        loss.backward()
        optimizer.step()

    for i, label in enumerate(word_list):
        W, WT = model.parameters()
        x, y = W[0][i].item(), W[1][i].item()
        plt.scatter(x, y)
        plt.annotate(label, xy=(x, y), xytext=(5, 2),
                     textcoords='offset points', ha='right', va='bottom')
    plt.show()
```

### 结果展示

![image-20230927103723596](https://gitee.com/mianmann/drawing-bed-warehouse/raw/master/img/image-20230927103723596.png)

## NNLM

### 实战代码

```python
import torch
import torch.nn as nn
import torch.optim as optim


def make_batch():
    input_batch = []
    target_batch = []

    for sen in sentences:
        word = sen.split()  # space tokenizer
        input = [word_dict[n] for n in word[:-1]]  # create (1~n-1) as input
        target = word_dict[word[-1]]  # create (n) as target

        input_batch.append(input)
        target_batch.append(target)

    return input_batch, target_batch


# Model
class NNLM(nn.Module):
    def __init__(self):
        super(NNLM, self).__init__()
        self.C = nn.Embedding(n_class, m)
        self.H = nn.Linear(n_step * m, n_hidden, bias=False)
        self.d = nn.Parameter(torch.ones(n_hidden))
        self.U = nn.Linear(n_hidden, n_class, bias=False)
        self.W = nn.Linear(n_step * m, n_class, bias=False)
        self.b = nn.Parameter(torch.ones(n_class))

    def forward(self, X):
        X = self.C(X)  # X: [batch_size, n_step, m]
        X = X.view(-1, n_step * m) # [batch_size, n_step * m]
        tanh = torch.tanh(self.d + self.H(X)) # [batch_size, n_hidden]
        output = self.b + self.W(X) + self.U(tanh) # [batch_size, n_class]
        return output
        


if __name__ == '__main__':
    n_step = 2  # number of steps, n-1 in paper
    n_hidden = 2  # number of hidden size, h in paper
    m = 2  # embedding size, m in paper

    sentences = ["i like dog", "i love coffee", "i hate milk"]

    word_list = " ".join(sentences).split()
    word_list = list(set(word_list))
    word_dict = {w: i for i, w in enumerate(word_list)}
    number_dict = {i: w for i, w in enumerate(word_list)}
    n_class = len(word_list)

    model = NNLM()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    input_batch, target_batch = make_batch()
    input_batch = torch.LongTensor(input_batch)
    target_batch = torch.LongTensor(target_batch)

    # Training
    for epoch in range(5000):
        optimizer.zero_grad()
        # input_batch: [batch_size, n_step, m]
        # 这里为 [[0,5], [0,2], [0,6]]
        output = model(input_batch)

        # output: [batch_size,n_class], target_batch: [batch_size]
        loss = criterion(output, target_batch)
        if (epoch + 1) % 1000 == 0:
            print(f'Epoch: {epoch + 1}, loss: {loss}')

        loss.backward()
        optimizer.step()

    # Predict
    predict = model(input_batch).data.max(1, keepdim=True)[1]

    # Test
    print([sen.split()[:2] for sen in sentences], '->',
          [number_dict[n.item()] for n in predict.squeeze()])

```

### 结果展示

![image-20230927191612488](https://gitee.com/mianmann/drawing-bed-warehouse/raw/master/img/image-20230927191612488.png)

### 其他

附上NNLM参照的讲解https://zhuanlan.zhihu.com/p/565343505

## TextCNN实战

### 实战代码

```python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(self):
        super(TextCNN, self).__init__()
        self.num_filters_total = num_filters * len(filter_sizes)
        self.W = nn.Embedding(vocab_size, embedding_size)
        self.Weight = nn.Linear(self.num_filters_total, num_classes, bias=False)
        self.Bias = nn.Parameter(torch.ones([num_classes]))
        self.filter_list = nn.ModuleList(
            [nn.Conv2d(1, num_filters, (size, embedding_size)) for size in filter_sizes])

    def forward(self, X):
        # [batch_size, sequence_length, embedding_length]
        embedded_chars = self.W(X)
        # add channel(=1) [batch_size, channel(=1), sequence_length, embedding_size]
        embedded_chars = embedded_chars.unsqueeze(1)

        pooled_outputs = []
        for i, conv in enumerate(self.filter_list):
            # conv: [input_channel(=1), output_channel(=3), (filter_height, filter_width), bias_option]
            h = F.relu(conv(embedded_chars))
            # mp: ((filter_height, filter_width))
            mp = nn.MaxPool2d((sequence_length - filter_sizes[i] + 1, 1))
            # pooled: [batch_size(=6), output_height(=1), output_width(=1), output_channel(=3)]
            pooled = mp(h).permute(0, 2, 3, 1)
            pooled_outputs.append(pooled)

        # [batch_size(=6), output_height(=1), output_width(=1), output_channel(=3) * 3]
        h_pool = torch.cat(pooled_outputs, len(filter_sizes))
        # [batch_size(=6), output_weight(=1), output_width(=1), output_channel(=3)]
        h_pool_flat = torch.reshape(h_pool, [-1, self.num_filters_total])
        # [batch_size, num_classes]
        model = self.Weight(h_pool_flat) + self.Bias
        return model

if __name__ == '__main__':
    embedding_size = 2
    sequence_length = 3
    num_classes = 2 # nums of target classes
    filter_sizes = [2, 2, 2]  # n-gram windows
    num_filters = 3

    # 3 words sentences (sequence_length=3)
    sentences = ["i love you", "he loves me", "she likes baseball",
                 "i hate you", "sorry for that", "this is awful"]
    labels = [1, 1, 1, 0, 0, 0]  # 1 is good, 0 is not good.

    word_list = " ".join(sentences).split()
    word_list = list(set(word_list))
    word_dict = {w: i for i, w in enumerate(word_list)}
    vocab_size = len(word_list)

    model = TextCNN()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    inputs = torch.LongTensor(
        [np.asarray([word_dict[n] for n in sen.split()]) for sen in sentences])
    targets = torch.LongTensor([out for out in labels])

    # Training
    for epoch in range(5000):
        optimizer.zero_grad()
        output = model(inputs)

        # output: [batch_size, num_classes], target_batch: [batch_size]
        loss = criterion(output, targets)
        if (epoch + 1) % 1000 == 0:
            print(f'Epoch: {epoch + 1}, Loss: {loss}')

        loss.backward()
        optimizer.step()

    # Test
    test_text = 'sorry hate you'
    tests = [np.asarray([word_dict[n] for n in test_text.split()])]
    test_batch = torch.LongTensor(tests)

    # Predict
    predict = model(test_batch).data.max(1, keepdim=True)[1]
    if predict[0][0] == 0:
        print(test_text, "is Bad Mean...")
    else:
        print(test_text, "is Good Mean!!")
```

### 结果展示

![](https://gitee.com/mianmann/drawing-bed-warehouse/raw/master/img/image-20231002185744681.png)

### 代码讲解

下图是TextCNN是流程图，接下来我会跟着这张图来分析我们的代码

![TextCNN 详解| LuoTeng's Blog](https://i.loli.net/2019/12/09/CVwojkfiMnTchpZ.png)

代码演示的是TextCNN用于文本分类任务的实战，其中需要注意的是filter_sizes是卷积核的宽度，即为流程图中的region_sizes，（这里是[2,2,2]是因为只是作简单演示，在流程图中写的是[2,3,4]）；然后num_filters即为流程图中每一个region的filter个数（流程图中是2，代码中是3）。从流程图中可以看到，首先进行Embedding，这里对应forward中的第一步，然后需要添加一个维度作为通道维度；再经过卷积层，可以得到由卷积核遍历嵌入矩阵得到的左三，这里对应代码中的conv操作，可以看到代码中是初始化了一个Conv2d的列表（用于存储不同宽度的卷积核），然后遍历该列表，执行每个卷积核的操作，将卷积结果由ReLU激活；从左三到左四是最大池化的过程，对应代码中的MaxPool2d，然后调整结果维度顺序，将通道维度放在最后面；从左四到左五对应代码中的cat和reshape过程；从左五到最后对应代码中的最后的线性函数，将长度为num_filter_total（图中是6，代码中是9）的结果向量转化为长度为num_classes的长度向量，及我们需要的分类得分情况（图中最后还使用了softmax函数进行正则化，相当于模型输出的直接是概率了）

## TextRNN实战

### 实战代码

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


def make_batch():
    input_batch = []
    target_batch = []

    for sen in sentences:
        word = sen.split()
        input = [word_dict[n] for n in word[:-1]]
        target = word_dict[word[-1]]

        input_batch.append(np.eye(n_class)[input])
        target_batch.append(target)

    return input_batch, target_batch


class TextRNN(nn.Module):
    def __init__(self):
        super(TextRNN, self).__init__()
        self.rnn = nn.RNN(input_size=n_class, hidden_size=n_hidden)
        self.W = nn.Linear(n_hidden, n_class, bias=False)
        self.b = nn.Parameter(torch.ones([n_class]))

    def forward(self, hidden, X):
        X = X.transpose(0, 1)  # X: [n_step, batch_size, n_class]
        outputs, hidden = self.rnn(X, hidden)
        # outputs : [n_step, batch_size, num_directions(=1) * n_hidden]
        # hidden : [num_layers(=1) * num_directions(=1), batch_size, n_hidden]
        outputs = outputs[-1]  # [batch_size, num_directions(=1) * n_hidden] (这里使用hidden[-1]也可以)
        model = self.W(outputs) + self.b  # model : [batch_size, n_class]
        return model


if __name__ == '__main__':
    n_step = 2  # number of cells(=number of step)
    n_hidden = 5  # number of hidden units in one cell

    sentences = ["i like dog", "i love coffee", "i hate milk"]

    word_list = " ".join(sentences).split()
    word_list = list(set(word_list))
    word_dict = {w: i for i, w in enumerate(word_list)}
    number_dict = {i: w for i, w in enumerate(word_list)}
    n_class = len(word_dict)
    batch_size = len(sentences)

    model = TextRNN()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    input_batch, target_batch = make_batch()
    input_batch = torch.FloatTensor(input_batch)
    target_batch = torch.LongTensor(target_batch)

    # Training
    for epoch in range(5000):
        optimizer.zero_grad()

        # hidden : [num_layers * num_directions, batch, hidden_size]
        hidden = torch.zeros(1, batch_size, n_hidden)
        # input_batch : [batch_size, n_step, n_class]
        output = model(hidden, input_batch)

        # output : [batch_size, n_class], target_batch : [batch_size]
        loss = criterion(output, target_batch)
        if (epoch + 1) % 1000 == 0:
            print('Epoch:', '%04d' % (epoch + 1),
                  'cost =', '{:.6f}'.format(loss))

        loss.backward()
        optimizer.step()

    input = [sen.split()[:2] for sen in sentences]

    # Predict
    hidden = torch.zeros(1, batch_size, n_hidden)
    predict = model(hidden, input_batch).data.max(1, keepdim=True)[1]
    print([sen.split()[:2] for sen in sentences], '->',
          [number_dict[n.item()] for n in predict.squeeze()])
```

### 结果展示

![](https://gitee.com/mianmann/drawing-bed-warehouse/raw/master/img/image-20231003151630067.png)

### 代码讲解

下图是RNN的基本框架图，下面我将依照此讲解代码

![image-20230911151645445](https://gitee.com/mianmann/drawing-bed-warehouse/raw/master/img/image-20230911151645445.png)

代码大部分地方处理都大同小异，这里仅对nn.RNN作部分关键讲解：RNN中input_size即为输入的词向量矩阵宽度（输入层节点个数），hidden_size即为隐藏层节点个数，num_layers即为RNN层数（多层RNN），bidirectional表示是否为双向RNN；nn.RNN的返回值有两个，第一个是RNN网络所有时间步的隐藏状态的集合$y^{(t)}$，第二个是最后一个时间步的隐藏状态$h^{(t)}$

(nn.RNN的讲解参照https://blog.csdn.net/lwgkzl/article/details/88717678，通俗易懂）

## TextLSTM实战

### 实战代码

```python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def make_batch():
    input_batch, target_batch = [], []

    for seq in seq_data:
        # 'm','a','k' is input
        input = [word_dict[n] for n in seq[:-1]]
        # 'e' is target
        target = word_dict[seq[-1]]
        input_batch.append(np.eye(n_class)[input])
        target_batch.append(target)

    return input_batch, target_batch


class TextLSTM(nn.Module):
    def __init__(self):
        super(TextLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=n_class, hidden_size=n_hidden)
        self.W = nn.Linear(n_hidden, n_class, bias=False)
        self.b = nn.Parameter(torch.ones([n_class]))

    def forward(self, X):
        # X: [n_step, batch_size, n_class]
        input = X.transpose(0, 1)
        # [num_layers(=1) * num_directions(=1), batch_size, n_hidden]
        hidden_state = torch.zeros(1, len(X), n_hidden)
        cell_state = torch.zeros(1, len(X), n_hidden)

        outputs, (_, _) = self.lstm(input, (hidden_state, cell_state))
        outputs = outputs[-1]  # [batch_size, n_hidden]
        model = self.W(outputs) + self.b  # model : [batch_size, n_class]
        return model


if __name__ == '__main__':
    n_step = 3  # number of cells(= number of Step)
    n_hidden = 128  # number of hidden units in one cell

    char_arr = [c for c in 'abcdefghijklmnopqrstuvwxyz']
    word_dict = {n: i for i, n in enumerate(char_arr)}
    number_dict = {i: w for i, w in enumerate(char_arr)}
    n_class = len(word_dict)  # number of class(=number of vocab)

    seq_data = ['make', 'need', 'coal', 'word',
                'love', 'hate', 'live', 'home', 'hash', 'star']

    model = TextLSTM()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    input_batch, target_batch = make_batch()
    input_batch = torch.FloatTensor(input_batch)
    target_batch = torch.LongTensor(target_batch)

    # Training
    for epoch in range(1000):
        optimizer.zero_grad()

        output = model(input_batch)
        loss = criterion(output, target_batch)
        if (epoch + 1) % 100 == 0:
            print('Epoch:', '%04d' % (epoch + 1),
                  'cost =', '{:.6f}'.format(loss))

        loss.backward()
        optimizer.step()

    inputs = [sen[:3] for sen in seq_data]

    predict = model(input_batch).data.max(1, keepdim=True)[1]
    print(inputs)
    print('->', [number_dict[n.item()] for n in predict.squeeze()])
```

### 结果展示

![image-20231003203703585](https://gitee.com/mianmann/drawing-bed-warehouse/raw/master/img/image-20231003203703585.png)

### 代码讲解

![image-20230912130422464](https://gitee.com/mianmann/drawing-bed-warehouse/raw/master/img/image-20230912130422464.png)

LSTM是RNN的一种，所以与TextRNN基本相同，主要特点就是多了一个Cell层用于存储长期记忆，所以有三个输入值和返回值（第三个对应cell层）

## BiLSTM实战

### 实战代码

```python
import numpy as np
import torch
import torch.nnsas nn
import torch.optim as optim


def make_batch():
    input_batch = []
    target_batch = []

    words = sentence.split()
    for i, word in enumerate(words[:-1]):
        input = [word_dict[n] for n in words[:(i + 1)]]
        input = input + [0] * (max_len - len(input))
        target = word_dict[words[i + 1]]
        input_batch.append(np.eye(n_class)[input])
        target_batch.append(target)

    return input_batch, target_batch


class BiLSTM(nn.Module):
    def __init__(self):
        super(BiLSTM, self).__init__()

        self.lstm = nn.LSTM(input_size=n_class,
                            hidden_size=n_hidden, bidirectional=True)
        self.W = nn.Linear(n_hidden * 2, n_class, bias=False)
        self.b = nn.Parameter(torch.ones([n_class]))

    def forward(self, X):
        input = X.transpose(0, 1)  # input : [n_step, batch_size, n_class]

        # [num_layers(=1) * num_directions(=2), batch_size, n_hidden]
        hidden_state = torch.zeros(1*2, len(X), n_hidden)
        # [num_layers(=1) * num_directions(=2), batch_size, n_hidden]
        cell_state = torch.zeros(1*2, len(X), n_hidden)

        outputs, (_, _) = self.lstm(input, (hidden_state, cell_state))
        outputs = outputs[-1]  # [batch_size, n_hidden]
        model = self.W(outputs) + self.b  # model : [batch_size, n_class]
        return model


if __name__ == '__main__':
    n_hidden = 5  # number of hidden units in one cell

    sentence = (
        'Lorem ipsum dolor sit amet consectetur adipisicing elit '
        'sed do eiusmod tempor incididunt ut labore et dolore magna '
        'aliqua Ut enim ad minim veniam quis nostrud exercitation'
    )

    word_dict = {w: i for i, w in enumerate(list(set(sentence.split())))}
    number_dict = {i: w for i, w in enumerate(list(set(sentence.split())))}
    n_class = len(word_dict)
    max_len = len(sentence.split())

    model = BiLSTM()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    input_batch, target_batch = make_batch()
    input_batch = torch.FloatTensor(input_batch)
    target_batch = torch.LongTensor(target_batch)

    # Training
    for epoch in range(10000):
        optimizer.zero_grad()
        output = model(input_batch)
        loss = criterion(output, target_batch)
        if (epoch + 1) % 1000 == 0:
            print('Epoch:', '%04d' % (epoch + 1),
                  'cost =', '{:.6f}'.format(loss))

        loss.backward()
        optimizer.step()

    predict = model(input_batch).data.max(1, keepdim=True)[1]
    print(sentence)
    print([number_dict[n.item()] for n in predict.squeeze()])
```

### 结果展示

![image-20231003230311579](https://gitee.com/mianmann/drawing-bed-warehouse/raw/master/img/image-20231003230311579.png)

### 代码分析

这是一个文本生成任务，与TextLSTM的区别为设置了bidirectional=True，所以为双向LSTM，至于这里为什么要将双向LSTM用于文本生成我也不是很理解。需要注意的是我们在make_batch时，有一步添加[0]站位符的处理，这是因为在文本生成任务中，模型需要接受固定长度的输入序列，所以不足的地方加以填充

## Seq2Seq

### 实战代码

```python
import numpy as np
import torch
import torch.nn as nn

# S: Symbol that shows starting of decoding input
# E: Symbol that shows starting of decoding output
# P: Symbol that will fill in blank sequence if current batch data size is short than time steps


def make_batch():
    input_batch, output_batch, target_batch = [], [], []

    for seq in seq_data:
        for i in range(2):
            seq[i] = seq[i] + 'P' * (n_step - len(seq[i]))

        input = [num_dic[n] for n in seq[0]]
        output = [num_dic[n] for n in ('S' + seq[1])]
        target = [num_dic[n] for n in (seq[1] + 'E')]

        input_batch.append(np.eye(n_class)[input])
        output_batch.append(np.eye(n_class)[output])
        target_batch.append(target)  # not one-hot

    # make tensor
    return torch.FloatTensor(input_batch), torch.FloatTensor(output_batch), torch.LongTensor(target_batch)


# make test batch
def make_testbatch(input_word):
    input_batch, output_batch = [], []

    input_w = input_word + 'P' * (n_step - len(input_word))
    input = [num_dic[n] for n in input_w]
    output = [num_dic[n] for n in 'S' + 'P' * n_step]

    input_batch = np.eye(n_class)[input]
    output_batch = np.eye(n_class)[output]

    return torch.FloatTensor(input_batch).unsqueeze(0), torch.FloatTensor(output_batch).unsqueeze(0)


# Model
class Seq2Seq(nn.Module):
    def __init__(self):
        super(Seq2Seq, self).__init__()
        self.enc_cell = nn.RNN(
            input_size=n_class, hidden_size=n_hidden, dropout=0.5)
        self.dec_cell = nn.RNN(
            input_size=n_class, hidden_size=n_hidden, dropout=0.5)
        self.fc = nn.Linear(n_hidden, n_class)

    def forward(self, enc_input, enc_hidden, dec_input):
        # enc_input: [max_len(=n_step, time step), batch_size, n_class]
        enc_input = enc_input.transpose(0, 1)
        # dec_input: [max_len(=n_step, time step), batch_size, n_class]
        dec_input = dec_input.transpose(0, 1)

        # enc_states : [num_layers(=1) * num_directions(=1), batch_size, n_hidden]
        _, enc_states = self.enc_cell(enc_input, enc_hidden)
        # outputs : [max_len+1(=6), batch_size, num_directions(=1) * n_hidden(=128)]
        outputs, _ = self.dec_cell(dec_input, enc_states)

        model = self.fc(outputs)  # model : [max_len+1(=6), batch_size, n_class]
        return model


if __name__ == '__main__':
    n_step = 5
    n_hidden = 128

    char_arr = [c for c in 'SEPabcdefghijklmnopqrstuvwxyz']
    num_dic = {n: i for i, n in enumerate(char_arr)}
    seq_data = [['man', 'women'], ['black', 'white'], ['king', 'queen'], [
        'girl', 'boy'], ['up', 'down'], ['high', 'low']]

    n_class = len(num_dic)
    batch_size = len(seq_data)

    model = Seq2Seq()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    input_batch, output_batch, target_batch = make_batch()

    for epoch in range(5000):
        # make hidden shape [num_layers * num_directions, batch_size, n_hidden]
        hidden = torch.zeros(1, batch_size, n_hidden)

        optimizer.zero_grad()
        # input_batch : [batch_size, max_len(=n_step, time step), n_class]
        # output_batch : [batch_size, max_len+1(=n_step, time step) (because of 'S' or 'E'), n_class]
        # target_batch : [batch_size, max_len+1(=n_step, time step)], not one-hot
        output = model(input_batch, hidden, output_batch)
        # output : [max_len+1, batch_size, n_class]
        output = output.transpose(0, 1)  # [batch_size, max_len+1(=6), n_class]
        loss = 0
        for i in range(0, len(target_batch)):
            # output[i] : [max_len+1, n_class, target_batch[i] : max_len+1]
            loss += criterion(output[i], target_batch[i])
        if (epoch + 1) % 1000 == 0:
            print('Epoch:', '%04d' % (epoch + 1),
                  'cost =', '{:.6f}'.format(loss))
        loss.backward()
        optimizer.step()

    # Test

    def translate(word):
        input_batch, output_batch = make_testbatch(word)

        # make hidden shape [num_layers * num_directions, batch_size, n_hidden]
        hidden = torch.zeros(1, 1, n_hidden)
        output = model(input_batch, hidden, output_batch)
        
        # output : [max_len+1(=6), batch_size(=1), n_class]
        predict = output.data.max(2, keepdim=True)[1]  # select n_class dimension
        decoded = [char_arr[i] for i in predict]
        end = decoded.index('E')
        translated = ''.join(decoded[:end])

        return translated.replace('P', '')


    print('test')
    print('man ->', translate('man'))
    print('mans ->', translate('mans'))
    print('king ->', translate('king'))
    print('black ->', translate('black'))
    print('upp ->', translate('upp'))
```

### 结果展示

![image-20231006223340668](https://gitee.com/mianmann/drawing-bed-warehouse/raw/master/img/image-20231006223340668.png)

### 代码分析

(1)先来简单说一下Seq2Seq模型，让我们从RNN开始，RNN的结构有三种：

- N to N：该模型处理的一般是输入和输出序列长度相等的任务（如词性标注、语言模型）
- 1 to N：此类结构的输入长度为1，输出长度为N，一般又可以分为两种：一种是将输入只输入到第一个神经元，另一种将输入输入到所有神经元（如图像生成文字）
- N to 1：和1 to N相反（如情感分类）

①可以看到，RNN结构大多对序列的长度比较局限，对于类似于机器翻译的任务，输入和输出长度并不对等，为N to M的结构，简单的RNN束手无策，因此便有了新的模型，Encoder-Decoder模型，也就是Seq2Seq模型

②模型一般由两部分组成：第一部分是Encoder部分，用于对输入的N长度的序列进行表征；第二部分是Decoder部分，用于将Encoder提取出的表征建立起到输出的M长度序列的映射。一般情况下两个都是N to N的RNN网络，M长度序列的映射是Encoder的隐藏状态，会被用作Decoder的隐藏层

(2)Seq2Seq模型的关键点就在于Encoder和Decoder，下面我们将根据代码来讲解最容易疑惑的几个问题

- 为什么使用"S"+"目标序列"来构成output字符序列？并且在训练模型时将该序列用于decoder的输入？
  答：对于第一个问题，S是开始的意思，使用S作为开始的token可以使我们的目标字符序列右移一个时间步，因为模型在每个时间步都尝试生成下一个输出符号（这一步会使用到上一个时间步的输出），所以在机器翻译的实际情况中我们需要一个start_token去开始生成我们的结果；对于第二问题，原因也是同上，这里的处理被称为**Teacher Forcing**（当某一个单元输出出错时，如果将其输出传递给下一个单元，可能导致错误传递下去。这时候，需要在一定程度上减少这种传递，就采用按一定的比例决定是否对神经单元采用上一个上一个单元的输出作为输入），在训练模型时使用正确的output可以增加公式中正确部分的比例权重，帮助引导模型产生最为理想的权重（因为最为理想的情况就是上一个时间步的生成是正确的），但这也会出现未来泄露的问题（遮蔽未来的方法详见Attention）
- 为什么使用Encoder的隐藏状态来作为Decoder的隐藏层？而不是Encoder的输出？
  答：因为Encoder的隐藏层状态包含了输入序列的抽象表示，可以看作是输入序列的"语义信息"或"上下文信息"，对于Decoder来说，这个信息是非常有用的，可以帮助它更好地生成输出序列；而Encoder的输出已经被线性函数转化成了概率，是不包含这些信息的，所以使用Encoder的隐藏状态来作为Decoder的隐藏层
- 从第一个问题就能看出make_testbatch为什么使用"S"+"P"*作为dec_input了，因为这是实际情况，我们没有未来信息

## Attention实战(Seq2Seq)

### 实战代码

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# S: Symbol that shows starting of decoding input
# E: Symbol that shows starting of decoding output
# P: Symbol that will fill in blank sequence if current batch data size is short than time steps


def make_batch():
    input_batch = [np.eye(n_class)[[word_dict[n]
                                    for n in sentences[0].split()]]]
    output_batch = [np.eye(n_class)[[word_dict[n]
                                     for n in sentences[1].split()]]]
    target_batch = [[word_dict[n] for n in sentences[2].split()]]

    # make tensor
    return torch.FloatTensor(input_batch), torch.FloatTensor(output_batch), torch.LongTensor(target_batch)


class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.enc_cell = nn.RNN(
            input_size=n_class, hidden_size=n_hidden, dropout=0.5)
        self.dec_cell = nn.RNN(
            input_size=n_class, hidden_size=n_hidden, dropout=0.5)

        # Linear for attention
        self.attn = nn.Linear(n_hidden, n_hidden)
        self.out = nn.Linear(n_hidden * 2, n_class)

    def forward(self, enc_inputs, hidden, dec_inputs):
        # enc_inputs: [n_step(=n_step, time step), batch_size, n_class]
        enc_inputs = enc_inputs.transpose(0, 1)
        # dec_inputs: [n_step(=n_step, time step), batch_size, n_class]
        dec_inputs = dec_inputs.transpose(0, 1)

        # enc_outputs : [n_step, batch_size, num_directions(=1) * n_hidden], matrix F
        # enc_hidden : [num_layers(=1) * num_directions(=1), batch_size, n_hidden]
        enc_outputs, enc_hidden = self.enc_cell(enc_inputs, hidden)

        trained_attn = []
        hidden = enc_hidden
        n_step = len(dec_inputs)
        model = torch.empty([n_step, 1, n_class])

        for i in range(n_step):  # each time step
            # dec_output : [n_step(=1), batch_size(=1), num_directions(=1) * n_hidden]
            # hidden : [num_layers(=1) * num_directions(=1), batch_size(=1), n_hidden]
            dec_output, hidden = self.dec_cell(
                dec_inputs[i].unsqueeze(0), hidden)
            attn_weights = self.get_att_weight(
                dec_output, enc_outputs)  # attn_weights : [1, 1, n_step]
            trained_attn.append(attn_weights.squeeze().data.numpy())

            # matrix-matrix product of matrices [1,1,n_step] x [1,n_step,n_hidden] = [1,1,n_hidden]
            context = attn_weights.bmm(enc_outputs.transpose(0, 1))
            # dec_output : [batch_size(=1), num_directions(=1) * n_hidden]
            dec_output = dec_output.squeeze(0)
            context = context.squeeze(1)  # [1, num_directions(=1) * n_hidden]
            model[i] = self.out(torch.cat((dec_output, context), 1))

        # make model shape [n_step, n_class]
        return model.transpose(0, 1).squeeze(0), trained_attn

    # get attention weight one 'dec_output' with 'enc_outputs'
    def get_att_weight(self, dec_output, enc_outputs):
        n_step = len(enc_outputs)
        attn_scores = torch.zeros(n_step)  # attn_scores : [n_step]

        for i in range(n_step):
            attn_scores[i] = self.get_att_score(dec_output, enc_outputs[i])

        # Normalize scores to weights in range 0 to 1
        return F.softmax(attn_scores).view(1, 1, -1)

    # enc_outputs [batch_size, num_directions(=1) * n_hidden]
    def get_att_score(self, dec_output, enc_output):
        score = self.attn(enc_output)  # score : [batch_size, n_hidden]
        # inner product make scalar value
        return torch.dot(dec_output.view(-1), score.view(-1))


if __name__ == '__main__':
    n_step = 5  # number of cells(= number of Step)
    n_hidden = 128  # number of hidden units in one cell

    sentences = ['ich mochte ein bier P', 'S i want a beer', 'i want a beer E']

    word_list = " ".join(sentences).split()
    word_list = list(set(word_list))
    word_dict = {w: i for i, w in enumerate(word_list)}
    number_dict = {i: w for i, w in enumerate(word_list)}
    n_class = len(word_dict)  # vocab list

    # hidden : [num_layers(=1) * num_directions(=1), batch_size, n_hidden]
    hidden = torch.zeros(1, 1, n_hidden)

    model = Attention()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    input_batch, output_batch, target_batch = make_batch()

    # Train
    for epoch in range(2000):
        optimizer.zero_grad()
        output, _ = model(input_batch, hidden, output_batch)

        loss = criterion(output, target_batch.squeeze(0))
        if (epoch + 1) % 400 == 0:
            print('Epoch:', '%04d' % (epoch + 1),
                  'cost =', '{:.6f}'.format(loss))

        loss.backward()
        optimizer.step()

    # Test
    test_batch = [np.eye(n_class)[[word_dict[n] for n in 'SPPPP']]]
    test_batch = torch.FloatTensor(test_batch)
    predict, trained_attn = model(input_batch, hidden, test_batch)
    predict = predict.data.max(1, keepdim=True)[1]
    print(sentences[0], '->', [number_dict[n.item()]
          for n in predict.squeeze()])

    # Show Attention
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(trained_attn, cmap='viridis')
    ax.set_xticklabels([''] + sentences[0].split(), fontdict={'fontsize': 14})
    ax.set_yticklabels([''] + sentences[2].split(), fontdict={'fontsize': 14})
    plt.show()
```

### 结果展示

(1)训练结果：

![image-20231009210445295](https://gitee.com/mianmann/drawing-bed-warehouse/raw/master/img/image-20231009210445295.png)

(2)注意力热力图：

![image-20231009210518200](https://gitee.com/mianmann/drawing-bed-warehouse/raw/master/img/image-20231009210518200.png)

### 代码分析

因为nn.RNN返回的第一个参数就是每个时间步的隐藏状态的集合，所以不要被这里使用enc_output和dec_output计算attention误解；然后还要注意的一点是这里decoder中的计算变成了遍历每个时间步分别计算，因为要计算每个时间步的attention_weight，当这样时，我们就需要手动更新hidden（对应遍历部分的第一行代码）

## Transformer实战

### 实战代码

```python
import math
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import matplotlib.pyplot as plt


def make_batch(sentences):
    enc_inputs, dec_inputs, dec_outputs = [], [], []

    for i in range(len(sentences)):
        enc_input = [src_vocab[n] for n in sentences[i][0].split()]
        dec_input = [tgt_vocab[n] for n in sentences[i][1].split()]
        dec_output = [tgt_vocab[n] for n in sentences[i][2].split()]

        enc_inputs.append(enc_input)
        dec_inputs.append(dec_input)
        dec_outputs.append(dec_output)

    return torch.LongTensor(enc_inputs), torch.LongTensor(dec_inputs), torch.LongTensor(dec_outputs)


def get_attn_pad_mask(seq_q, seq_k):
    """
    pad mask的作用: 在对value向量加权平均的时候, 可以让pad对应的alpha_ij=0, 这样注意力就不会考虑到pad向量
    这里的q,k表示的是两个序列 (和注意力机制的q,k没有关系)
    encoder和decoder都有可能调用这个函数, 所以seq_len视情况而定, 两个seq_len不一定相等
    seq_q: [batch_size, seq_len]
    seq_k: [batch_size, seq_len]
    """

    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()

    # eq(zero) is PAD token
    # pad_attn_mask: [batch_size, 1, seq_len]
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)

    # return [batch_size, seq_len, seq_len]
    return pad_attn_mask.expand(batch_size, len_q, len_k)


def get_attn_subsequence_mask(seq):
    """
    subsequence mask的作用: 在decoder的self-attention中, 为了防止decoder看到未来的信息, 需要将未来的信息遮掩
    seq: [batch_size, tgt_len]
    """
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]

    # 生成一个上三角矩阵, 1表示未来的信息, 0表示当前和过去的信息
    # subsequence_mask: [batch_size, tgt_len, tgt_len]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1)
    # 这里为什么要转换成byte类型?
    subsequence_mask = torch.from_numpy(subsequence_mask)
    return subsequence_mask


class MyDataSet(Data.Dataset):
    """自定义的Dataset"""

    def __init__(self, enc_inputs, dec_inputs, dec_outputs):
        super(MyDataSet, self).__init__()
        self.enc_inputs = enc_inputs
        self.dec_inputs = dec_inputs
        self.dec_outputs = dec_outputs

    def __len__(self):
        return self.enc_inputs.shape[0]

    def __getitem__(self, idx):
        return self.enc_inputs[idx], self.dec_inputs[idx], self.dec_outputs[idx]


class PositionalEncoding(nn.Module):
    """Transformer中的位置编码"""

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        # 因为div_term是序列，所以pos矩阵需要添加一个维度来与其匹配相乘
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # 下面的过程即为原论文中的pe计算公式
        div_term = torch.exp(torch.arange(0, d_model, 2).float()
                             * (-math.log(10000.0)) / d_model)

        # 原论文中计算公式2i+1和2i均对应的2i，所以这里均为div_term
        # 偶数维度
        pe[:, 0::2] = torch.sin(position * div_term)
        # 奇数维度
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # [src_len, batch_size, d_model] + [src_len, 1, d_model]
        x = x + self.pe[:x.size(0), :]
        # 在位置编码中使用dropout能增加模型的鲁棒性，防止过拟合
        return self.dropout(x)


class ScaledDotProductAttention(nn.Module):
    """对注意力进行缩放点积"""

    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, Q, K, V, attn_mask):
        """
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        """

        # scores: [batch_size, n_heads, len_q, len_k]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
        # 用mask矩阵填充scores (用-1e9填充scores中与attn_mask中值为1位置相对应的元素)
        scores.masked_fill_(attn_mask, -1e9)
        # 对最后一个维度进行softmax，即对每个词计算attention权重
        # attn: [batch_size, n_heads, len_q, len_k]
        attn = self.softmax(scores)
        # context: [batch_size, n_heads, len_q, d_v]
        context = torch.matmul(attn, V)
        return context, attn


class MultiHeadAttention(nn.Module):
    """
    这个Attention类可以实现:
    Encoder的Self-attention
    Decoder的Masked Self-attention
    Encoder-Decoder的Attention(Decoder的第二个Attention)
    """

    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        # Q,K维度必须相同，不然无法做点积
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)

    def forward(self, input_Q, input_K, input_V, attn_mask):
        """
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        attn_mask: [batch_size, seq_len, seq_len]
        """
        residual, batch_size = input_Q, input_Q.size(0)

        # 下面的多头参数矩阵是放在一起做线性变换的，然后再拆成多个头，这是工程实现的技巧
        # Q: [batch_size, n_heads, len_q, d_k]
        Q = self.W_Q(input_Q).view(
            batch_size, -1, n_heads, d_k).transpose(1, 2)
        # K: [batch_size, n_heads, len_k, d_k]
        K = self.W_K(input_K).view(
            batch_size, -1, n_heads, d_k).transpose(1, 2)
        # V: [batch_size, n_heads, len_v(=len_k), d_v]
        V = self.W_V(input_V).view(
            batch_size, -1, n_heads, d_v).transpose(1, 2)

        # 因为是多头，所以mask矩阵要扩充成4维的
        # attn_mask: [batch_size, seq_len, seq_len] -> [batch_size, n_heads, seq_len, seq_len]
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)

        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context, attn = ScaledDotProductAttention()(Q, K, V, attn_mask)
        # 下面将不同头的输出向量拼接在一起
        # context: [batch_size, len_q, n_heads * d_v] -> [batch_size, len_q, d_model]
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, n_heads * d_v)

        # 最后再做一个projection
        # output: [batch_size, len_q, d_model]
        output = self.fc(context)

        # LayerNorm和Residual即为论文中的Add&Norm步骤
        return nn.LayerNorm(d_model).to(device)(output + residual), attn


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        # d_ff: 2048
        # 在Transformer中，bias通常不是必须的，这里可以试一下添加bias看看效果是否更好
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False)
        )

    def forward(self, inputs):
        # inputs: [batch_size, seq_len, d_model]

        residual = inputs
        # output: [batch_size, seq_len, d_model]
        output = self.fc(inputs)

        return nn.LayerNorm(d_model).to(device)(output + residual)


class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        # self-attention
        self.enc_self_attn = MultiHeadAttention()
        # feed forward
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs, enc_self_attn_mask):
        """
        EncoderLayer的前向传播
        enc_inputs: [batch_size, src_len, d_model]
        enc_self_attn_mask: [batch_size, src_len, src_len]
        """

        # 第一个enc_inputs是Q，第二个enc_inputs是K，第三个enc_inputs是V
        # enc_outputs: [batch_size, src_len, d_model], enc_self_attn: [batch_size, n_heads, src_len, src_len]
        enc_outputs, enc_self_attn = self.enc_self_attn(
            enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
        # enc_outputs: [batch_size, src_len, d_model]
        enc_outputs = self.pos_ffn(enc_outputs)

        return enc_outputs, enc_self_attn


class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        # self-attention
        self.dec_self_attn = MultiHeadAttention()
        # encoder-decoder attention
        self.dec_enc_attn = MultiHeadAttention()
        # feed forward
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        """
        dec_inputs: [batch_size, tgt_len, d_model]
        enc_inputs: [batch_size, src_len, d_model]
        enc_outputs: [batch_size, src_len, d_model]
        dec_self_attn_mask: [batch_size, tgt_len, tgt_len]
        dec_enc_attn_mask: [batch_size, tgt_len, src_len]
        """

        # dec_outputs: [batch_size, tgt_len, d_model], dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len]
        dec_outputs, dec_self_attn = self.dec_self_attn(
            dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        # dec_outputs: [batch_size, tgt_len, d_model], dec_enc_attn: [batch_size, n_heads, tgt_len, src_len]
        dec_outputs, dec_enc_attn = self.dec_enc_attn(
            dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)

        # dec_outputs: [batch_size, tgt_len, d_model]
        dec_outputs = self.pos_ffn(dec_outputs)

        return dec_outputs, dec_self_attn, dec_enc_attn


class Encoder(nn.Module):
    """ Transformer编码器"""

    def __init__(self):
        super(Encoder, self).__init__()
        # token Embedding
        self.src_emb = nn.Embedding(src_vocab_size, d_model)
        # Transformer中的位置编码是固定的，不需要学习
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

    def forward(self, enc_inputs):
        # enc_inputs: [batch_size, src_len]
        # enc_outputs: [batch_size, src_len, d_model]
        enc_outputs = self.src_emb(enc_inputs)
        # enc_outputs: [batch_size, src_len, src_len]
        enc_outputs = self.pos_emb(enc_outputs.transpose(0, 1)).transpose(0, 1)
        # Encoder输入序列的pad mask矩阵
        # enc_self_attn_mask: [batch_size, src_len, src_len]
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)
        # 用来保存返回的attention的值，以便最后画热力图，用来看各个词之间的关系
        enc_self_attns = []
        for layer in self.layers:
            # 上一个block的enc_outputs为当前block的输入
            # enc_ouputs: [batch_size, src_len, d_model], enc_self_attn: [batch_size, n_heads, src_len, src_len]
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)

        return enc_outputs, enc_self_attns


class Decoder(nn.Module):
    """Transformer解码器"""

    def __init__(self):
        super(Decoder, self).__init__()
        # token Embedding
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)
        # Transformer中的位置编码是固定的，不需要学习
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])

    def forward(self, dec_inputs, enc_inputs, enc_outputs):
        """
        dec_inputs: [batch_size, tgt_len]
        enc_inputs: [batch_size, src_len]
        enc_outputs: [batch_size, src_len, d_model]
        """
        dec_outputs = self.tgt_emb(dec_inputs)
        dec_outputs = self.pos_emb(
            dec_outputs.transpose(0, 1)).transpose(0, 1).to(device)

        # Decoder的Masked Multi-head Attention
        # dec_self_attn_pad_mask: [batch_size, tgt_len, tgt_len] (pad mask)
        dec_self_attn_pad_mask = get_attn_pad_mask(
            dec_inputs, dec_inputs).to(device)
        # dec_self_attn_subsequence_mask: [batch_size, tgt_len, tgt_len] (future mask)
        dec_self_attn_subsequence_mask = get_attn_subsequence_mask(
            dec_inputs).to(device)

        # 两个mask矩阵相加,既屏蔽了pad，又屏蔽了未来信息
        dec_self_attn_mask = torch.gt(
            (dec_self_attn_pad_mask + dec_self_attn_subsequence_mask), 0).to(device)

        # Decoder的Encoder-Decoder Multihead Attention
        # dec_self_attn_mask: [batch_size, tgt_len, tgt_len]
        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs).to(device)

        # 用来保存返回的attention的值，以便最后画热力图，用来看各个词之间的关系
        dec_self_attns, dec_enc_attns = [], []

        for layer in self.layers:
            # 上一个block的dec_outputs为当前block的输入
            # dec_outputs: [batch_size, tgt_len, d_model], dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len]
            # dec_enc_attn: [batch_size, n_heads, tgt_len, src_len]
            dec_outputs, dec_self_attn, dec_enc_attn = layer(
                dec_outputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)

        return dec_outputs, dec_self_attns, dec_enc_attns


class Transformer(nn.Module):
    """Transformer总网络"""

    def __init__(self):
        super(Transformer, self).__init__()
        self.encoder = Encoder().to(device)
        self.decoder = Decoder().to(device)
        self.projection = nn.Linear(
            d_model, tgt_vocab_size, bias=False).to(device)

    def forward(self, enc_inputs, dec_inputs):
        """
        Transformer的输入为两个序列
        enc_inputs: [batch_size, src_len]
        dec_inputs: [batch_size, tgt_len]
        """

        # enc_outputs: [batch_size, src_len, d_model]
        # enc_self_attns: [n_layer,batch_size,n_heads,src_len,src_len]
        enc_outputs, enc_self_attns = self.encoder(enc_inputs)

        # dec_outputs: [batch_size, tgt_len, d_model]
        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(
            dec_inputs, enc_inputs, enc_outputs)

        # dec_logits: [batch_size, tgt_len, tgt_vocab_size]
        dec_logits = self.projection(dec_outputs)

        return dec_logits.view(-1, dec_logits.size(-1)), enc_self_attns, dec_self_attns, dec_enc_attns


def greedy_decoder(model, enc_input, start_symbol):
    """
    用来生成翻译结果的函数
    enc_input: [batch_size, src_len]
    start_symbol: [S]
    """
    enc_outputs, enc_self_attns = model.encoder(enc_input)
    dec_input = torch.zeros(1, 0).type_as(enc_input.data)
    terminal = False
    next_symbol = start_symbol
    while not terminal:
        # 预测阶段：dec_input序列会一点点变长 (每次只预测一个词)
        dec_input = torch.cat(
            [dec_input.to(device), torch.tensor([[next_symbol]], dtype=enc_input.dtype).to(device)], dim=-1)
        dec_outputs, _, _ = model.decoder(
            dec_input, enc_input, enc_outputs)
        projected = model.projection(dec_outputs)
        prob = projected.squeeze(0).max(dim=-1, keepdim=False)[1]

        # 取出预测的最后一个词
        next_word = prob.data[-1]
        next_symbol = next_word

        # 如果预测的最后一个词是终止符, 则停止预测
        if next_symbol == tgt_vocab["E"]:
            terminal = True

    # 返回预测的序列
    greedy_dec_predict = dec_input[:, 1:]
    return greedy_dec_predict


def showgraph(attn, i, x, y):
    """
    用来生成注意力权重图的函数
    """

    # 这里只看多头注意力中的第一个注意力头
    attn = attn[-1][i][0]
    attn = attn.squeeze(0).data.numpy()
    fig = plt.figure(figsize=(n_heads, n_heads))
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attn, cmap='viridis')
    ax.set_xticks(range(len(x)))
    ax.set_yticks(range(len(y)))
    ax.set_xticklabels(x, fontdict={'fontsize': 14})
    ax.set_yticklabels(y, fontdict={'fontsize': 14})
    plt.show()


if __name__ == '__main__':
    # device = 'cuda'
    device = 'cpu'

    epochs = 100

    """
    S: Symbol that shows starting of decoding input
    E: Symbol that shows starting of decoding output
    P: Symbol that will fill in blank sequence if current batch data size is short than time steps
    """
    sentences = [
        # 德语和英语单词个数可以不相同
        # enc_input, dec_input, dec_output
        ['ich mochte ein bier P', 'S i want a beer .', 'i want a beer . E'],
        ['ich mochte ein cola P', 'S i want a coke .', 'i want a coke . E']
    ]

    # 德语和英语分开建立词库
    # padding should be zero
    src_vocab = {'P': 0, 'ich': 1, 'mochte': 2, 'ein': 3, 'bier': 4, 'cola': 5}
    src_number_dict = {i: w for i, w in enumerate(src_vocab)}
    src_vocab_size = len(src_vocab)

    tgt_vocab = {'P': 0, 'i': 1, 'want': 2, 'a': 3,
                 'beer': 4, 'coke': 5, '.': 6, 'S': 7, 'E': 8}
    tgt_number_dict = {i: w for i, w in enumerate(tgt_vocab)}
    tgt_vocab_size = len(tgt_vocab)

    src_len = 5  # max sequence length of enc_input
    tgt_len = 6  # max sequence length of dec_input

    # Transformer Parameters
    d_model = 512  # Embedding Size
    d_ff = 2048  # FeedForward dimension
    d_k = d_v = 64  # dimension of Q、K、V
    n_layers = 6  # number of Encoder and Decoder blocks
    n_heads = 8  # number of heads in Multi-Head Attention

    enc_inputs, dec_inputs, dec_outputs = make_batch(sentences)

    # enc_inputs: [batch_size, src_len]
    # dec_inputs: [batch_size, tgt_len]
    # dec_outputs: [batch_size, tgt_len]
    dataset = MyDataSet(enc_inputs, dec_inputs, dec_outputs)
    loader = Data.DataLoader(dataset, 2, True)

    model = Transformer().to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    # 用Adam的话效果不好，可能是因为Transformer中的参数太多了
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.99)

    # Train
    for epoch in range(epochs):
        for enc_inputs, dec_inputs, dec_outputs in loader:
            # enc_inputs: [batch_size, src_len]
            # dec_inputs: [batch_size, tgt_len]
            # dec_outputs: [batch_size, tgt_len]

            enc_inputs, dec_inputs, dec_outputs = enc_inputs.to(
                device), dec_inputs.to(device), dec_outputs.to(device)

            # outputs: [batch_size * tgt_len, tgt_vocab_size]
            outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(
                enc_inputs, dec_inputs)

            loss = criterion(outputs, dec_outputs.view(-1))
            print('Epoch:', '%04d' % (epoch + 1),
                  'loss =', '{:.6f}'.format(loss))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Test
    enc_inputs, _, _ = next(iter(loader))
    for i in range(len(enc_inputs)):
        print(f'==================== Test Result {i+1} ====================')
        greedy_dec_predict = greedy_decoder(model, enc_inputs[i].view(
            1, -1).to(device), start_symbol=tgt_vocab["S"])
        print(enc_inputs[i], "->", greedy_dec_predict.squeeze())
        print([src_number_dict[t.item()] for t in enc_inputs[i]], '->',
              [tgt_number_dict[n.item()] for n in greedy_dec_predict.squeeze()])

        # Show Attention Graph
        print('first head of last state enc_self_attns')
        showgraph(enc_self_attns, i, [src_number_dict[n.item()] for n in enc_inputs[i]], [
                  src_number_dict[n.item()] for n in enc_inputs[i]])

        print('first head of last state dec_self_attns')
        showgraph(dec_self_attns, i, [tgt_number_dict[n.item()] for n in dec_inputs[i]], [
                  tgt_number_dict[n.item()] for n in dec_inputs[i]])

        print('first head of last state dec_enc_attns')
        showgraph(dec_enc_attns, i, [src_number_dict[n.item()] for n in enc_inputs[i]], [
                  tgt_number_dict[n.item()] for n in dec_inputs[i]])
```

### 结果展示

![](https://gitee.com/mianmann/drawing-bed-warehouse/raw/master/img/image-20231021144642391.png)

### 代码分析

自己光分析透彻这个代码就花了两天，这里实在没精力再重新过一遍并附上讲解了，望谅解(´ο｀*))

但是自己还有个疑问没有解决，就是损失函数如果使用Adam，模型的损失函数会直接过早收敛，然后效果很差（可以说基本没效果）如果有答案的话请告诉我~
