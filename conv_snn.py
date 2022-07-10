import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from spikingjelly.clock_driven import neuron, surrogate, functional

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
BATCH_SIZE = 64		# 每批处理数据的数量
EPOCHS = 10				# 数据集训练的轮次
LEARNING_RATE = 1e-3   # 学习率
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')		# 使用gpu还是cpu
# DEVICE = 'cpu'


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

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)


######################### SNN Model ###############################
class Model_2(nn.Module):
    def __init__(self, tau, T, v_threshold=1.0, v_reset=0.0):
        super().__init__()
        self.T = T

        # 进行卷积减少特征维数并且减少时间上的循环
        self.static_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
        )

        # 脉冲编码
        self.conv = nn.Sequential(
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            nn.MaxPool2d(2, 2),  # 14 * 14

            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            nn.MaxPool2d(2, 2)  # 7 * 7
        )

        # 两层全连接网络
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 32 * 4 * 4, bias=False),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            nn.Linear(32 * 4 * 4, 10, bias=False),# 因为有十个分类，所以输出为10
            neuron.IFNode(surrogate_function=surrogate.ATan()),
        )

    def forward(self, x):
        x = self.static_conv(x)# 不参与时间累加的卷积

        out_spikes_counter = self.fc(self.conv(x))# 进行编码
        for t in range(1, self.T):
            out_spikes_counter += self.fc(self.conv(x))

        return out_spikes_counter / self.T


def train(model, device, train_loader, optimizer, epoch, writer):
    model.train()

    for index, (img, label) in enumerate(train_loader):
        img, label = img.to(device), label.to(device)
        label_one_hot = F.one_hot(label, 10).float()

        optimizer.zero_grad()
        output = model(img)
        loss = F.mse_loss(output, label_one_hot)
        loss.backward()
        optimizer.step()

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
    model = Model_2(tau=2.0, T=2).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    writer = SummaryWriter(log_path)

    for epoch in tqdm(range(EPOCHS)):
        train(model, DEVICE, train_dataloader, optimizer, epoch, writer)
        test(model, DEVICE, test_dataloader, epoch, writer)
        writer.flush()

    # save model
    # torch.save(model, output_path+'/MNIST.pth')
    writer.close()
