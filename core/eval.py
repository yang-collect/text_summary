from datasets import load_metric
import torch
import torch.nn.functional as F
from transformers import MT5ForConditionalGeneration, BertTokenizer
import jieba
import os
import sys

abs_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(abs_path)

from lib import config, utils


def batch_decode(input_ids, tokenizer):
    return [tokenizer.decode(ids) for ids in input_ids]


class T5PegasusTokenizer(BertTokenizer):
    def __init__(self, pre_tokenizer=lambda x: jieba.cut(x, HMM=False), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pre_tokenizer = pre_tokenizer

    def _tokenize(self, text, *arg, **kwargs):
        split_tokens = []
        for text in self.pre_tokenizer(text):
            if text in self.vocab:
                split_tokens.append(text)
            else:
                split_tokens.extend(super()._tokenize(text))
        return split_tokens


def evaluate(model, eval_dataloader,tokenizer):
    """
    根据传入模型以及数据集计算accuracy
    """
    # 加载accuracy评估器
    metric = load_metric(config.score_path)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # 模型评估过程
    model.eval()
    with torch.no_grad():
        for batch in eval_dataloader:
            data = {k: v.to(device) for k, v in batch.items()}
            out = model(**data)
            prob = F.softmax(out.logits, dim=-1)
            # 对模型输出取argmax
            predictions = torch.argmax(prob, dim=-1)
            # 将当前批次数据的预测结果和原始结果传递给评估器
            metric.add_batch(predictions=batch_decode(predictions,tokenizer), references=batch_decode(batch['labels'],tokenizer))

    # 返回评估其计算结果
    return metric.compute()


if __name__ == '__main__':
    # print(config.score_path)
    # metric = load_metric(config.score_path)
    # print(metric)
    # 模型保存路径
    model_path = config.save_path
    # 加载预训练模型
    model = MT5ForConditionalGeneration.from_pretrained(model_path)
    # tokenizer实例化
    tokenizer = T5PegasusTokenizer.from_pretrained(str(model_path))
    # 训练即测试数据加载
    train_dataloder, test_dataloder = utils.DataLoad(tokenizer, config.train_path)

    # 打印训练的accuracy
    print('train data:', evaluate(model, train_dataloder,tokenizer))
    # 打印测试数据上的accuracy
    print('test data:', evaluate(model, test_dataloder,tokenizer))
    # # 加载评估数据
    # eval_dataloader = DataProcess.DataLoad(tokenizer, config.dev_path)
    # # 打印在评估数据上的accuracy
    # print('dev data', evaluate(model, eval_dataloader))
