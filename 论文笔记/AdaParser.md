### 挑战

① LLM 可能产生不准确的解析结果
② 对历史数据的依赖性较大
③ 速度效率问题

### 框架

![image-20240901181022876](https://gitee.com/mianmann/drawing-bed-warehouse/raw/master/img/image-20240901181022876.png)

模版解析树与 LILAC 大同小异

自生成的情境学习是 KNN 选择候选对

**模版修正器：**

主要针对以下两类错误进行修正，修正方式是 检查+Prompt
![image-20240901181742069](https://gitee.com/mianmann/drawing-bed-warehouse/raw/master/img/image-20240901181742069.png)

### 总结

创新点就是模版修正器，针对大模型幻觉现象容易出现的两类错误进行检查+提示修正