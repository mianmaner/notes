## 中文预处理

### 使用Bert

#### 1.导入Bert

```python
# 不同模型的分词方法是不同的
bert_tokenizer = BertTokenizer.from_pretrained(
    pretrained_model_name_or_path='bert-base-chinese',  # 下载基于 BERT 模型的分词方法的中文字典包
    cache_dir='./bert_vocab',  # 字典的下载位置
    force_download=False  # 不会重复下载
)
bert_model = BertModel.from_pretrained(
    pretrained_model_name_or_path='bert-base-chinese',
    cache_dir='./bert_model',
    force_download=False
)
```

#### 2.加载数据集

```python
dataset = pd.read_csv('./waimai_10k.csv', encoding='utf-8')
sentences = dataset['review'].tolist()
labels = dataset['label'].tolist()
```

#### 3.分词

```python
input_ids = []
attention_masks = []
token_type_ids = []
for i in tqdm(range(len(sentences)), total=len(sentences), desc='Tokenize'):
    encoded = bert_tokenizer.encode_plus(
        sentences[i],
        max_length=260,
        pad_to_max_length=True,
        add_special_tokens=True,
        truncation=True
    )
    input_ids.append(encoded['input_ids'])
    attention_masks.append(encoded['attention_mask'])
    token_type_ids.append(encoded['token_type_ids'])
```

#### 4.嵌入

```python
input_ids = torch.tensor(input_ids)
attention_masks = torch.tensor(attention_masks)
token_type_ids = torch.tensor(token_type_ids)

bert_model.eval()
with torch.no_grad():
    outputs = bert_model(
        input_ids, attention_mask=attention_masks, token_type_ids=token_type_ids)

embed_sentences = outputs.last_hidden_state
```

### 