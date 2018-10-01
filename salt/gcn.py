from lightai.imps import *
from .block import *


class GCN(nn.Module):
    def __init__(self, in_c, out_c, k):
        super().__init__()
        k = k if k%2==1 else k-1
        self.conv1 = ConvBlock(in_c, out_c, kernel_size=(k, 1), padding=(k//2, 0))
        self.conv2 = ConvBlock(out_c, out_c, kernel_size=(1, k), padding=(0, k//2))
        self.conv3 = ConvBlock(in_c, out_c, kernel_size=(1, k), padding=(0, k//2))
        self.conv4 = ConvBlock(out_c, out_c, kernel_size=(k, 1), padding=(k//2, 0))

    def forward(self, x):
        x1 = self.conv2(self.conv1(x))
        x2 = self.conv4(self.conv3(x))
        return x1 + x2 + x


class BR(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.conv1 = ConvBlock(in_c, in_c, kernel_size=3, padding=1)
        self.conv2 = ConvBlock(in_c, in_c, kernel_size=3, padding=1)

    def forward(self, x):
        origin = x
        out = self.conv2(self.conv1(x))
        return origin + out
