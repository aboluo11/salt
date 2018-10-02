from lightai.imps import *
from .block import *


class Attention(nn.Module):
    def __init__(self, in_c, key_c, value_c):
        super().__init__()
        self.key_c = key_c
        self.value_c = value_c
        self.key = ConvBlock(in_c, key_c, kernel_size=1, activation='leaky')
        self.query = ConvBlock(in_c, key_c, kernel_size=1, activation='leaky')
        # self.value = ConvBlock(in_c, value_c, kernel_size=1)
        # self.conv1 = ConvBlock(value_c, in_c, kernel_size=1)

    def forward(self, x):
        bs, channel, height, width = x.shape

        w1 = self.key(x).view(bs, self.key_c, -1)
        w2 = self.query(x).view(bs, self.key_c, -1)
        w = torch.matmul(w1.permute(0, 2, 1), w2)
        w = w / self.key_c**0.5
        w = F.softmax(w, dim=-1)

        value = x.view(bs, channel, -1).permute(0, 2, 1)

        out = torch.matmul(w, value)
        out = out.permute(0, 2, 1).view(bs, channel, height, width)

        # out = self.conv1(out)

        return out


class ObjectContext(nn.Module):
    def __init__(self, in_c, key_c, value_c, out_c):
        super().__init__()
        self.attention = Attention(in_c, key_c, value_c)
        self.conv1 = ConvBlock(2*in_c, out_c, kernel_size=1)

    def forward(self, x):
        context = self.attention(x)
        out = torch.cat([x, context], dim=1)
        out = self.conv1(out)
        return out
