from pathlib import Path

path = Path(__file__).parent.parent

data_path = path.joinpath('./data/nlpcc_data.json')
train_path = path.joinpath('./data/sample.json')
title_length = 60  # title的长度

content_length = 450  # content的长度

batch_size = 32

model_path = r'C:\Users\wie\Documents\pretrain_model\chinese_t5_pegasus_small'

epochs = 6

save_path = path.parent.joinpath('./model_file/summary')

num_warmup_steps = 500

score_path = str(path.parent.joinpath('./metrics/rouge.py'))
# print(score_path)
