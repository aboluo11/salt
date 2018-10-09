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
    def __init__(self, in_c, r=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = ConvBlock(in_c, in_c//r, kernel_size=1)
        self.conv2 = nn.Conv2d(in_c//r, in_c, kernel_size=1)

    def forward(self, x):
        origin = x
        x = self.pool(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.sigmoid(x)
        x = x * origin
        return x


class SpatialGate(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, 1, kernel_size=1)

    def forward(self, x):
        origin = x
        x = self.conv1(x)
        # x = x / (self.in_c**0.5)
        x = torch.sigmoid(x)
        x = x * origin
        return x


class SCBlock(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.spatial_gate = SpatialGate(in_c)
        self.channel_gate = ChannelGate(in_c)
        self.bn = nn.BatchNorm2d(in_c)

    def forward(self, x):
        x1 = self.spatial_gate(x)
        x2 = self.channel_gate(x)
        x = x1 + x2
        x = self.bn(x)
        return x