import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from spikingjelly.clock_driven import neuron, surrogate, functional,encoding

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from datetime import datetime

data_path = "data"
output_path = "output/SNN"
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_path = "/log/"+timestamp
os.makedirs(log_path,exist_ok=True)
os.makedirs(output_path,exist_ok=True)
######################### Hyperparameter ############################
BATCH_SIZE = 512		# 每批处理数据的数量
EPOCHS = 10				# 数据集训练的轮次
LEARNING_RATE = 1e-3   # 学习率
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')		# 使用gpu还是cpu


########################## Datasets #################################
train_dataset = datasets.MNIST(
    root = data_path,
    train=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081, ))
    ]),
    download=True
)
test_dataset = datasets.MNIST(
    root = data_path,
    train=False,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]),
    download=True
)

train_dataloader = DataLoader(train_dataset, batch_size=512, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=512, shuffle=True)


######################### SNN Model ###############################
class Model_1(nn.Module):
    def __init__(self, tau, T, v_threshold=1.0, v_reset=0.0):
        super().__init__()
        self.T = T
        self.encoder = encoding.PoissonEncoder()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 10, bias=False),
            neuron.LIFNode(tau=tau)
        )

    def forward(self, x):
        # 运行T个时长，out_spikes_counter是shape=[batch_size, 10]的tensor
        # 记录整个仿真时长内，输出层的10个神经元的脉冲发放次数
        for t in range(self.T):
            if t == 0:
                out_spikes_counter = self.net(self.encoder(x).float())
            else:
                out_spikes_counter += self.net(self.encoder(x).float())
        
        # out_spikes_counter / T 得到输出层10个神经元在仿真时长内的脉冲发放频率
        return out_spikes_counter / self.T


def train(model, device, train_loader, optimizer, epoch, writer):
    model.train()

    for index, (img, label) in enumerate(train_loader):
        img, label = img.to(device), label.to(device)
        label_one_hot = F.one_hot(label, 10).float()

        optimizer.zero_grad()
        output = model(img)
        # 损失函数为输出层神经元的脉冲发放频率，与真实类别的MSE
        # 这样的损失函数会使，当类别i输入时，输出层中第i个神经元的脉冲发放频率趋近1，而其他神经元的脉冲发放频率趋近0        
        loss = F.mse_loss(output, label_one_hot)
        loss.backward()
        optimizer.step()

        # 优化一次参数后，需要重置网络的状态，因为SNN的神经元是有“记忆”的
        functional.reset_net(model)

        writer.add_scalar('train loss', loss, index + epoch * len(train_loader))

def test(model, device, test_loader, epoch, writer):
    model.eval()
    test_loss = 0
    correct = 0
    for index, (img, label) in enumerate(test_loader):
        img, label = img.to(device), label.to(device)
        label_one_hot = F.one_hot(label, 10).float()

        output = model(img)
        functional.reset_net(model)

        test_loss += F.mse_loss(output, label_one_hot)
        pred = output.max(1, keepdim=True)[1]       # 找到概率最大的下标
        correct += pred.eq(label.view_as(pred)).sum().item()

    test_loss /= len(test_loader)
    writer.add_scalar('test loss', test_loss, epoch)
    accuracy = 100. * correct / len(test_loader.dataset)
    writer.add_scalar('accuracy',accuracy,epoch)

if __name__ == '__main__':
    model = Model_1(tau=2.0, T=100).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    writer = SummaryWriter(log_path)

    for epoch in tqdm(range(EPOCHS)):
        train(model, DEVICE, train_dataloader, optimizer, epoch, writer)
        test(model, DEVICE, test_dataloader, epoch, writer)
        writer.flush()

    # save model
    # torch.save(model, output_path+'/MNIST.pth')
    writer.close()
