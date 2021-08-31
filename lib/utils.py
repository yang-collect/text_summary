import json

from torch.utils.data import Dataset, DataLoader

from lib import config


def read_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return ([item['title'] for item in data][:20000], [item['content'] for item in data][:20000]), \
           ([item['title'] for item in data][20000:], [item['content'] for item in data][20000:])


class MyDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.title, self.content = data
        self.tokenizer = tokenizer
        self.length = len(self.title)
        self.title_ids = self.encode(self.title, config.title_length)
        self.content_ids = self.encode(self.content, config.content_length)

    def encode(self, text_list, max_length):
        input_ids = self.tokenizer(text_list,
                                   max_length=max_length,
                                   padding='max_length',
                                   truncation=True,
                                   is_split_into_words=True,  # 是否分词
                                   return_tensors='pt').input_ids
        return input_ids

    def __getitem__(self, index):
        return {'input_ids': self.content_ids[index], 'labels': self.title_ids[index]}

    def __len__(self):
        return self.length


def DataLoad(tokenizer, path=config.train_path):
    train_data, test_data = read_data(path)
    train_dataset = MyDataset(train_data, tokenizer)
    test_dataset = MyDataset(test_data, tokenizer)
    return DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True), \
           DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True)



# if __name__ == '__main__':
#     from transformers import MT5ForConditionalGeneration, T5Tokenizer

#     model_path = r'C:\Users\wie\Documents\pretrain_model\mt5-base'
#     model = MT5ForConditionalGeneration.from_pretrained(model_path)
#     tokenizer = T5Tokenizer.from_pretrained(model_path)
#     print(next(iter(DataLoad(tokenizer))))
    # print(data[data.keys()[0]])
