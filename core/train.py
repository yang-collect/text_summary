import torch
import numpy as np
from transformers import MT5ForConditionalGeneration, BertTokenizer, AdamW, get_scheduler
import jieba
import os
import sys

abs_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(abs_path)

from lib import config, utils



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


def train(train_path=config.data_path, model_path=config.model_path, epochs=config.epochs,
          save_path=config.save_path):
    # 加载预训练模型和tokenizer
    model = MT5ForConditionalGeneration.from_pretrained(model_path)
    tokenizer = T5PegasusTokenizer.from_pretrained(model_path)

    # 加载训练和测试集
    train_dataloder, test_dataloder = utils.DataLoad(tokenizer, train_path)
    # test_dataloder = DataProcess.DataLoad(tokenizer, test_path)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # 实例化优化器
    optimizer = AdamW(model.parameters(), lr=5e-5)

    num_training_steps = epochs * len(train_dataloder)
    # warm up
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=config.num_warmup_steps,
        num_training_steps=num_training_steps
    )

    # 模型训练
    val_loss = np.inf
    model.to(device)
    model.train()

    for epoch in range(epochs):
        batch_loss = []
        for num, batch in enumerate(train_dataloder):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss
            batch_loss.append(loss.item())
            # 梯度更新
            loss.backward()
            # 优化器和学习率更新
            optimizer.step()
            lr_scheduler.step()
            # 梯度清零
            optimizer.zero_grad()
            # 每100个打印一次结果
            if num % 5000 == 0:
                print(f'epoch:{epoch},batch :{num} ,train_loss :{loss} !')

        epoch_loss = np.mean(batch_loss)
        avg_val_loss = compute_loss(model, test_dataloder)
        print(f'epoch:{epoch},tran_loss:{epoch_loss},valid loss;{avg_val_loss}')
        print('*' * 100)
        # Update minimum evaluating loss.
        if avg_val_loss < val_loss:
            tokenizer.save_pretrained(config.save_path)
            model.save_pretrained(config.save_path)
            val_loss = avg_val_loss

    print(val_loss)


def compute_loss(model, val_data):
    """Evaluate the loss and f1 score for an epoch.

    Args:
        model (torch.nn.Module): The model to evaluate.
        val_data (dataset.PairDataset): The evaluation data set.

    Returns:
        numpy ndarray: The average loss of the dev set.
    """
    print('validating')
    # metric = load_metric("f1")
    # metric = load_metric("f1")
    val_loss = []
    with torch.no_grad():
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        for batch, data in enumerate(val_data):
            data = {k: v.to(device) for k, v in data.items()}
            out = model(**data)
            loss = out.loss
        val_loss.append(loss.item())

    return np.mean(val_loss)


if __name__ == '__main__':
    train()
