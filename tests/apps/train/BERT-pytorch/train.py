import sys
import ctypes
import os
import time

from torch.utils.data import DataLoader

from bert_pytorch.model import BERT
from bert_pytorch.trainer import BERTTrainer
from bert_pytorch.dataset import BERTDataset, WordVocab

script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)

# load remoting bottom library
remoting_bottom_lib_path = os.getenv('REMOTING_BOTTOM_LIBRARY')
if remoting_bottom_lib_path is not None:
    cpp_lib = ctypes.CDLL(remoting_bottom_lib_path)
    start_trace = cpp_lib.startTrace
    end_trace = cpp_lib.endTrace

log_breakpoint_lib_path = os.getenv('LOG_BREAKPOINT_LIBRARY')
if log_breakpoint_lib_path is not None:
    cpp_lib = ctypes.CDLL(log_breakpoint_lib_path)
    breakpoint = cpp_lib.log_breakpoint

if(len(sys.argv) < 3):
    print('Usage: python3 train.py num_iter batch_size')
    sys.exit()

num_iter = int(sys.argv[1])
batch_size = int(sys.argv[2])

train_dataset = script_dir + "/data/corpus.small"
test_dataset = None
vocab_path = script_dir + "/data/vocab.small"
output_path = script_dir + "/output/bert.model"

hidden = 256
layers = 8
attn_heads = 8
seq_len = 20

num_workers = 0

with_cuda = True
log_freq = 10
corpus_lines = None
cuda_devices = None
on_memory = True

lr = 1e-3
adam_weight_decay = 0.01
adam_beta1 = 0.9
adam_beta2 = 0.999

os.makedirs(script_dir + "/output", exist_ok=True)

print("Loading Vocab", vocab_path)
vocab = WordVocab.load_vocab(vocab_path)
print("Vocab Size: ", len(vocab))

print("Loading Train Dataset", train_dataset)
train_dataset = BERTDataset(train_dataset, vocab, seq_len=seq_len,
                            corpus_lines=corpus_lines, on_memory=on_memory)

print("Loading Test Dataset", test_dataset)
test_dataset = BERTDataset(test_dataset, vocab, seq_len=seq_len, on_memory=on_memory) \
    if test_dataset is not None else None

print("Creating Dataloader")
train_data_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers)
test_data_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers) \
    if test_dataset is not None else None

print("Building BERT model")
bert = BERT(len(vocab), hidden=hidden, n_layers=layers, attn_heads=attn_heads)

print("Creating BERT Trainer")
trainer = BERTTrainer(bert, len(vocab), train_dataloader=train_data_loader, test_dataloader=test_data_loader,
                        lr=lr, betas=(adam_beta1, adam_beta2), weight_decay=adam_weight_decay,
                        with_cuda=with_cuda, cuda_devices=cuda_devices, log_freq=log_freq)

print("Training Start")

trainer.iterative_train(2)

if log_breakpoint_lib_path is not None:
    breakpoint()

if remoting_bottom_lib_path is not None:
    start_trace()
    
print("begin trace")
sys.stdout.flush()

T1 = time.time()
    
trainer.iterative_train(num_iter)
        
T2 = time.time()
print('time used: ', T2-T1)

if remoting_bottom_lib_path is not None:
    end_trace()
