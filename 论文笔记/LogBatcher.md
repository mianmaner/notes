### 挑战

① 基于语法的日志解析器性能不好，基于语义的日志解析器训练开销大
② 基于 LLM 的日志解析器过度依赖演示，且调用成本高

### 框架

![image-20240716171749677](https://gitee.com/mianmann/drawing-bed-warehouse/raw/master/img/image-20240716171749677.png)

#### 分区

tokenization＋vectorization（使用 TF-IDF 进行加权求和）得到每条日志的向量表示，然后 DBSCAN 聚类＋排序

#### 缓存

单纯的模版缓存机制，没有用树结构

#### 分批查询

分区后同区内相似度高的日志为一个批次进行查询，然后匹配和更新缓存

### 总结

无监督的 llm 日志解析器范式
