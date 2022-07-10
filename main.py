import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from spikingjelly.clock_driven import neuron, surrogate, functional,encoding

from tqdm import tqdm
from datetime import datetime

import argparse
import wandb
from model import *

import logging
def logger_config(log_path,logging_name):
    '''
    配置log
    :param log_path: 输出log路径
    :param logging_name: 记录中name，可随意
    :return:
    '''
    '''
    logger是日志对象，handler是流处理器，console是控制台输出（没有console也可以，将不会在控制台输出，会在日志文件中输出）
    '''
    # 获取logger对象,取名
    logger = logging.getLogger(logging_name)
    # 输出DEBUG及以上级别的信息，针对所有输出的第一层过滤
    logger.setLevel(level=logging.DEBUG)
    # 获取文件日志句柄并设置日志级别，第二层过滤
    handler = logging.FileHandler(log_path, encoding='UTF-8')
    handler.setLevel(logging.INFO)
    # 生成并设置文件日志格式
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    # console相当于控制台输出，handler文件输出。获取流句柄并设置日志级别，第二层过滤
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    console.setLevel(logging.DEBUG)
    # 为logger对象添加句柄
    logger.addHandler(handler)
    logger.addHandler(console)
    return logger
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_path = os.getcwd()+'/log/'+timestamp+'.log'
logger = logger_config(log_path,'test')

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size',type=int,default=128)
parser.add_argument('--epochs',type=int,default=10,help='how many epoch to run')
parser.add_argument('--lr',type=float,default=1e-3,help='the learning rate')
parser.add_argument('--device',default='cpu',help='cuda num or cpu')
parser.add_argument('--dataset',required=True,choices=['mnist','cifar10','fashionMnist'],help='the choice of dataset')
parser.add_argument('--optimizer',required=True,choices=['adam','rmsProp','adaGrad'])
parser.add_argument('--loss_func',required=True,choices=['mse','ce'])# TODO:再看下还有啥损失函数
parser.add_argument('--model',required=True,choices=['model0','model1','model2'])
parser.add_argument('--T',type=int,default=10,help='T in LIF model')
parser.add_argument('--wandb',default=False,help='whether to use wandb')
args = parser.parse_args()
if args.wandb:
    wandb.init(config=args,project='snn',resume=True)
logger.info(vars(args))

#-----------
# load dataset
#-----------
my_transforms=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081, ))
    ])
if args.dataset == 'mnist':
    train_dataset = datasets.MNIST(root='data',train=True,transform=my_transforms,download=True)
    test_dataset = datasets.MNIST(root='data',train=False,transform=my_transforms,download=True)
elif args.dataset == 'fashionMnist':
    train_dataset = datasets.FashionMNIST(root='data',train=True,transform=my_transforms,download=True)
    test_dataset = datasets.FashionMNIST(root='data',train=False,transform=my_transforms,download=True)
elif args.dataset == 'cifar10':
    train_dataset = datasets.CIFAR10(root='data',train=True,transform=my_transforms,download=True)
    test_dataset = datasets.CIFAR10(root='data',train=False,transform=my_transforms,download=True)
else:
    raise NotImplementedError(args.dataset+'is not implement')

train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

#--------------
# load model
#--------------
tau = 2.0
T = args.T
if args.model == 'model0':
    model = Model_0(tau=tau,T=T)
elif args.model == 'model1':
    model = Model_1(tau=tau,T=T)
elif args.model == 'model2':
    model = Model_2(tau=tau,T=T)
else:
    raise NotImplementedError(args.model+'is not implement')

model = model.to(args.device)
#-----------
# load optimizer & loss_fn
#-----------

# load optimizer
if args.optimizer == 'adam':
    optimizer = optim.Adam(model.parameters(),lr=args.lr)
elif args.optimizer == 'rmsProp':
    optimizer = optim.RMSprop(model.parameters(),lr=args.lr)
elif args.optimizer == 'adaGrad':
    optimizer = optim.Adagrad(model.parameters(),lr=args.lr)
else: 
    raise NotImplementedError(args.optimizer+'is not implement')

# load loss_func
if args.loss_func == 'mse':
    loss_fn = nn.MSELoss()
elif args.loss_func == 'ce':
    loss_fn = nn.CrossEntropyLoss()
else:
    raise NotImplementedError(args.loss_func+'is not implement')

#------------------
# train
#------------------
def train():
    model.train()
    correct = 0
    for i,(img,label) in enumerate(train_dataloader):
        img,label = img.to(args.device),label.to(args.device)
        label_one_hot = F.one_hot(label, 10).float()

        optimizer.zero_grad()
        output = model(img)
        # 损失函数为输出层神经元的脉冲发放频率，与真实类别的MSE
        # 这样的损失函数会使，当类别i输入时，输出层中第i个神经元的脉冲发放频率趋近1，而其他神经元的脉冲发放频率趋近0        
        loss = loss_fn(output, label_one_hot)
        loss.backward()
        optimizer.step()

        # 优化一次参数后，需要重置网络的状态，因为SNN的神经元是有“记忆”的
        functional.reset_net(model)    
        if args.wandb:
            wandb.log({'train_loss':loss})
        pred = output.max(1, keepdim=True)[1]       # 找到概率最大的下标
        correct += pred.eq(label.view_as(pred)).sum().item()
    accuracy = 100. * correct / len(train_dataloader.dataset)
    return accuracy

def test():
    model.eval()
    correct =0
    for i,(img,label) in enumerate(test_dataloader):
        img, label = img.to(args.device), label.to(args.device)
        label_one_hot = F.one_hot(label, 10).float()

        output = model(img)
        functional.reset_net(model)

        loss_fn(output, label_one_hot)
        pred = output.max(1, keepdim=True)[1]       # 找到概率最大的下标
        correct += pred.eq(label.view_as(pred)).sum().item()
    accuracy = 100. * correct / len(test_dataloader.dataset)
    return accuracy

if __name__ == '__main__':
    for e in tqdm(range(args.epochs)):
        train_accuracy = train()
        test_accuracy = test()
        if args.wandb:
            wandb.log({'train_accuracy':train_accuracy})
            wandb.log({'test_accuracy':test_accuracy})

