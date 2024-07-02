from mingpt.model import GPT
import torch
from torch.utils.data import Dataset
import fsspec
from dataclasses import dataclass
import sys
import ctypes
import os
import time

@dataclass
class DataConfig:
    path: str = None
    block_size: int = None      # 输入序列长度    
    vocab_size: int = None      # 词表尺寸
    train_split: float = None   # 训练集和测试集划分
    truncate: float = 1.0       # 用于训练的数据占全体数据的比例

class CharDataset(Dataset):

    def __init__(self, data_cfg: DataConfig): #data_path: str, block_size):
        # 加载所需比例的数据
        data = fsspec.open(data_cfg.path).open().read().decode('utf-8')
        data = data[ : int(len(data) * data_cfg.truncate)]

        # Set 去重，转 list 后排序得到数据集中的唯一字符列表作为词表
        chars = sorted(list(set(data))) 
        data_size = len(data)

        # 得到字符和词表索引之间的双射
        self.stoi = {ch: i for i, ch in enumerate(chars)}   # 字符 -> 词表索引
        self.itos = {i: ch for i, ch in enumerate(chars)}   # 词表索引 -> 字符
        
        self.block_size = data_cfg.block_size  	# 模型输入序列长度
        self.vocab_size = data_cfg.vocab_size			# 词表尺寸
        self.data = data

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        # grab a chunk of (block_size + 1) characters from the data
        chunk = self.data[idx:idx + self.block_size + 1]
        
        # encode every character to an integer
        dix = [self.stoi[s] for s in chunk]
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y

script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)

# load remoting bottom library
path = os.getenv('REMOTING_BOTTOM_LIBRARY')
if path is not None:
    cpp_lib = ctypes.CDLL(path)
    start_trace = cpp_lib.startTrace
    end_trace = cpp_lib.endTrace

if(len(sys.argv) < 3):
    print('Usage: python3 train.py num_iter batch_size')
    sys.exit()

num_iter = int(sys.argv[1])
batch_size = int(sys.argv[2])

model_config = GPT.get_default_config()
model_config.model_type = 'gpt2'
model_config.vocab_size = 50257 # openai's model vocabulary
model_config.block_size = 1024  # openai's model block_size (i.e. input context length)
model = GPT(model_config)
# your subclass of torch.utils.data.Dataset that emits example
# torch LongTensor of lengths up to 1024, with integers from [0,50257)
train_dataset = CharDataset(DataConfig(path=script_dir + '/data/input.txt', block_size=1024, vocab_size=50257))

if path is not None:
    start_trace()
    
T1 = time.time()

from mingpt.trainer import Trainer
train_config = Trainer.get_default_config()
train_config.learning_rate = 5e-4 # many possible options, see the file
train_config.max_iters = num_iter
train_config.batch_size = batch_size
trainer = Trainer(train_config, model, train_dataset)
trainer.run()

T2 = time.time()
print('time used: ', T2-T1)

if path is not None:
    end_trace()
