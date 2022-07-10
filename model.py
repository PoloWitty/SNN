import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from spikingjelly.clock_driven import neuron, surrogate, functional,encoding

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