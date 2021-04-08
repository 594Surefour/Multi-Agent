import torch
import torch.nn as nn
import torch.nn.functional as F
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


class Net(nn.Module):

    #输入输出层
    def __init__(self, input_size, output_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, output_size)
    #前向传播
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    #连接
    def hook(self, gradient):
        return 2*gradient

net = Net(input_size=1, output_size=1)
x = torch.linspace(-5, 5, 10).view(10, 1)
y = net(x)
