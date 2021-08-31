import json
from jieba import cut
import re
from lib import config

path = config.data_path


def clean_weibo_content(content: str):
    """
    对微博数据中的文本内容进行清洗
    Args:
        content: 文本

    Returns:

    """
    # 去除网址
    content = re.sub(r"(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b", "", content)
    # 合并正文中过多的空格
    content = re.sub(r"\s+", " ", content)
    # 去除\u200b字符
    content = content.replace("\u200b", "")
    return content


def read_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    s = []
    data = [{'title': list(cut(item['title'])), 'content': list(cut(clean_weibo_content(item['content'])))} for item in
            data]
    for item in data:
        if len(item['content']) <= 10 or len(item['title']) <= 5:
            continue
        if len(item['content']) <= 600:
            s.append(item)
    with open(config.train_path, 'w', encoding='utf-8') as f:
        json.dump(s, f, ensure_ascii=False)
    # title = [item['title'] for item in data if len(item['content']) <= 1300]
    # content = [item['content'] for item in data if len(item['content']) <= 1300]


if __name__ == '__main__':
    read_data(path)
    # print(len(s))
