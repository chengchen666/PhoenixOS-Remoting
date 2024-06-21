import time
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from utils.readData import read_dataset
from utils.ResNet import ResNet18
import sys
import ctypes
import os

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

# set device
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
# 读数据
train_loader,valid_loader,test_loader = read_dataset(batch_size=batch_size,pic_path='dataset')
# 加载模型(使用预处理模型，修改最后一层，固定之前的权重)
n_class = 10
model = ResNet18()
"""
ResNet18网络的7x7降采样卷积和池化操作容易丢失一部分信息,
所以在实验中我们将7x7的降采样层和最大池化层去掉,替换为一个3x3的降采样卷积,
同时减小该卷积层的步长和填充大小
"""
model.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
model.fc = torch.nn.Linear(512, n_class) # 将最后的全连接层改掉
model = model.to(device)
# 使用交叉熵损失函数
criterion = nn.CrossEntropyLoss().to(device)

# 开始训练
lr = 0.1
counter = 0
print("start training")

def train(iter_num):
    global train_loader
    global model
    global criterion
    global lr
    global counter
    
    iter = 0
    # keep track of training and validation loss
    train_loss = 0.0
    
    # 动态调整学习率
    if counter/10 ==1:
        counter = 0
        lr = lr*0.5
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    ###################
    # 训练集的模型 #
    ###################
    model.train() #作用是启用batch normalization和drop out
    while iter < iter_num:
        for data, target in train_loader:
            data = data.to(device)
            #print("after load data")
            target = target.to(device)
            #print("after load target")
            # clear the gradients of all optimized variables（清除梯度）
            optimizer.zero_grad()
            #print("after zero grad")
            # forward pass: compute predicted outputs by passing inputs to the model
            # (正向传递：通过向模型传递输入来计算预测输出)
            output = model(data).to(device)  #（等价于output = model.forward(data).to(device) ）
            #print("after forward")
            # calculate the batch loss（计算损失值）
            loss = criterion(output, target)
            # backward pass: compute gradient of the loss with respect to model parameters
            # （反向传递：计算损失相对于模型参数的梯度）
            loss.backward()
            #print("after backwarad")
            # perform a single optimization step (parameter update)
            # 执行单个优化步骤（参数更新）
            optimizer.step()
            # update training loss（更新损失）
            train_loss += loss.item()*data.size(0)
            # 计算平均损失
            train_loss = train_loss/len(train_loader.sampler)
            # 显示训练集与验证集的损失函数 
            print('Iter: {} \tTraining Loss: {:.6f}'.format(iter, train_loss))
            iter += 1
            if iter >= iter_num:
                break
    
    
train(2)

if path is not None:
    start_trace()
    
T1 = time.time()

train(num_iter)

T2 = time.time()
print(T2-T1)

if path is not None:
    end_trace()
