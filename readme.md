# 数据集介绍

数据是从`nlpcc`数据中进行分词并取其中词数小于600的作为训练和预测数据命名为sample，数据处理脚本详见`data\sample.py`

# 模型结果

模型的rouge评分如图所示：

具体脚本详见`core\eval.py` ，decode是对`greedy search`的结果进行decode

![image-20210901002307535](https://github.com/yang-collect/text_summary/blob/main/image-20210901002307535.png)

模型服务返回结果如下图：

生成采用的`beam search`，并未进行细致调参

![image-20210901005902763](https://github.com/yang-collect/text_summary/blob/main/image-20210901005902763.png)

# 模型介绍

预训练模型使用的是追一科技开源的[t5-pegasus pytorch small版本](https://github.com/renmada/t5-pegasus-pytorch)，以mT5为基础架构和初始权重，通过类似PEGASUS的方式进行预训练。

https://github.com/ZhuiyiTechnology/t5-pegasus



