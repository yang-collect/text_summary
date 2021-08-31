from flask import Flask, request, Response, jsonify
from concurrent.futures import ThreadPoolExecutor
from transformers import MT5ForConditionalGeneration, BertTokenizer
from jieba import cut
import json
import datetime
import torch

import os
import sys

abs_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(abs_path)

from lib import config


class MyResponse(Response):
    @classmethod
    def force_type(cls, response, environ=None):
        if isinstance(response, (list, dict)):
            response = jsonify(response)
        return super(Response, cls).force_type(response, environ)


# tokenizer 类
class T5PegasusTokenizer(BertTokenizer):
    def __init__(self, pre_tokenizer=lambda x: cut(x, HMM=False), *args, **kwargs):
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


# 创建服务
server = Flask(__name__)
# 约定flask接受参数的类型
server.response_class = MyResponse
# 创建一个线程池，默认线程数量为cpu核数的5倍
executor = ThreadPoolExecutor()
# fine-tune模型路径
model_path = config.save_path
# 加载模型
model = MT5ForConditionalGeneration.from_pretrained(model_path)
# tokenizer实例化
tokenizer = T5PegasusTokenizer.from_pretrained(model_path)


def stand_input(text):
    return list(cut(text))


# 绑定目录以及方法
@server.route('/text_classification/emotion_identify', methods=["POST"])
def scene_object_appearance_class():
    data = request.get_json()
    # print(data)
    output_res = {}
    if len(data) == 0:
        output_res["status"] = "400"
        output_res["msg"] = "Flase"
        output_res['text_label'] = "Flase"
        return output_res
    else:
        try:
            # 对传入的一条数据进行tokenizer
            token = tokenizer(stand_input(data['content']),
                              add_special_tokens=True,  # 添加special tokens， 也就是CLS和SEP
                              max_length=config.content_length,  # 设定最大文本长度
                              padding=True,  # padding
                              truncation=True,  # truncation
                              is_split_into_words=True,  # 是否分词
                              return_tensors='pt'  # 返回的类型为pytorch tensor
                              )
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            model.to(device)
            # print(token)
            out = model.generate(token['input_ids'],
                                 max_length=60,
                                 num_beams=5, # beam search的size大小
                                 no_repeat_ngram_size=2, # 确保一个二元组不至于出现两次
                                 early_stopping=True)[0]
            prediction = tokenizer.decode(out, skip_special_tokens=True).replace(' ', '')
            # 获取label对应的文本
            output_res['title'] = prediction
            return json.dumps(output_res, ensure_ascii=False)
        except Exception as e:
            print("异常原因: ", e)
            return {"error": 500}


def host():
    """ main 函数
    """
    HOST = '0.0.0.0'
    # 服务端口，为外部访问
    PORT = 5019
    server.config["JSON_AS_ASCII"] = False
    server.run(host=HOST, port=PORT, threaded=True)


if __name__ == "__main__":
    nowTime1 = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # 现在
    print("nowTime1: ", nowTime1)

    host()

    nowTime2 = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # 现在
    print("nowTime1: ", nowTime2)
