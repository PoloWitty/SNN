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
BATCH_SIZE = 512		# 每批处理数据的数量
EPOCHS = 10				# 数据集训练的轮次
LEARNING_RATE = 10e-3   # 学习率
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
class Model_0(nn.Module):
    def __init__(self, tau, T, v_threshold=1.0, v_reset=0.0):
        super().__init__()
        self.T = T

        self.conv = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=3, padding=1, bias=False),
            neuron.IFNode(v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan()),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(10, 20, kernel_size=3, padding=1, bias=False),
            neuron.IFNode(v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan()),
            nn.MaxPool2d(2, 2),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(20 * 7 * 7, 100, bias=False),
            neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan()),
            nn.Linear(100, 10, bias=False),
            neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan()),
        )

    def forward(self, x):
        x = self.conv(x)
        out_spike_counter = self.fc(x)

        for t in range(1, self.T):
            out_spike_counter += self.fc(x)

        return F.softmax(out_spike_counter / self.T)


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
    model = Model_0(tau=2.0, T=8).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    writer = SummaryWriter(log_path)

    for epoch in tqdm(range(EPOCHS)):
        train(model, DEVICE, train_dataloader, optimizer, epoch, writer)
        test(model, DEVICE, test_dataloader, epoch, writer)
        writer.flush()

    # save model
    # torch.save(model, output_path+'/MNIST.pth')
    writer.close()
