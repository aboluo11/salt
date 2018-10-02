from lightai.imps import *


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x):
        x = self.conv(x)
        x = torch.relu(x)
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
    def __init__(self, in_c):
        super().__init__()
        self.in_c = in_c
        self.conv1 = nn.Conv2d(in_c, 1, 1)
        # self.bn = nn.BatchNorm2d(in_c)

    def forward(self, x):
        origin = x
        x = self.conv1(x)
        x = x / (self.in_c**0.5)
        x = torch.sigmoid(x)
        x = x * origin
        # x = torch.relu(self.bn(x))
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