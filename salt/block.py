from lightai.core import *


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, stride=1, padding=0, activation='relu'):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        if activation == 'relu':
            self.activation = torch.relu
        elif activation == 'leaky':
            self.activation = nn.LeakyReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        x = self.bn(x)
        return x


class ChannelGate(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        r = 2
        self.linear1 = nn.Linear(in_c, in_c//r)
        self.linear2 = nn.Linear(in_c//r, in_c)
        self.bn1 = nn.BatchNorm1d(in_c//r)
        self.bn2 = nn.BatchNorm1d(in_c)

    def forward(self, x):
        bs = x.shape[0]
        origin = x
        x = x.view(*(x.shape[:2]), -1)
        x = torch.mean(x, dim=2)
        x = self.linear1(x)
        x = torch.relu(x)
        if bs > 1:
            x = self.bn1(x)
        x = self.linear2(x)
        if bs > 1:
            x = self.bn2(x)
        x = x.view(*x.shape, 1, 1)
        x = torch.sigmoid(x)
        x = x * origin
        return x


class SpatialGate(nn.Module):
    def __init__(self, in_c, writer, tag):
        super().__init__()
        self.conv1 = ConvBlock(in_c, in_c//2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_c//2, 1, kernel_size=3, padding=1)
        self.in_c = in_c
        self.writer = writer
        self.tag = f'{tag}_spatial_gate'

    def forward(self, x):
        origin = x
        x = self.conv1(x)
        x = self.conv2(x)
        # x = x / (self.in_c**0.5)
        x = torch.sigmoid(x)
        x = x * origin
        return x


class SCBlock(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.spatial_gate = SpatialGate(in_c)
        self.channel_gate = ChannelGate(in_c)

    def forward(self, x):
        x1 = self.spatial_gate(x)
        x2 = self.channel_gate(x)
        return x1 + x2