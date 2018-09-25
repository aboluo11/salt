from lightai.imps import *
from .log import *
from .unet import ChannelGate, SpatialGate

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, drop, layer_num, block_num, stride=1, downsample=None, writer=None):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        # self.drop = nn.Dropout2d(drop)
        self.writer = writer
        self.layer_num = layer_num
        self.block_num = block_num
        self.channel_gate = ChannelGate(planes)
        self.spatial_gate = SpatialGate(planes)

    def forward(self, x, global_step):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # out = self.drop(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        g1 = self.channel_gate(out)
        g2 = self.spatial_gate(out)
        out = (g1 + g2) * out

        if self.writer and self.block_num == 3:
            log_grad(writer=self.writer, model=self.conv2, tag=f'encoder_layer{self.layer_num}', global_step=global_step)

        return out


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, stride, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = torch.relu(x)
        return x


class MySequential(nn.Module):
    def __init__(self, *models):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x, global_step):
        for model in self.models:
            x = model(x, global_step)
        return x


class ResNet(nn.Module):
    def __init__(self, layers, block, drop, writer=None):
        self.inplanes = 32
        super().__init__()
        self.conv1 = ConvBlock(1, 32, 7, stride=1, padding=3)
        self.layer1 = self._make_layer(block, 64, layers[0], drop, writer, 1, stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], drop, writer, 2, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], drop, writer, 3, stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], drop, writer, 4, stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, n_blocks, drop, writer, layer_num, stride=1):
        downsample = None
        if stride != 1:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        blocks = []
        blocks.append(block(self.inplanes, planes, drop, layer_num, 1, stride, downsample, writer=writer))
        self.inplanes = planes
        for i in range(1, n_blocks):
            blocks.append(block(self.inplanes, planes, drop, layer_num, i+1, writer=writer))
        return MySequential(*blocks)

    def forward(self, x, global_step):
        x = self.conv1(x)
        x = self.layer1(x, global_step)
        x = self.layer2(x, global_step)
        x = self.layer3(x, global_step)
        x = self.layer4(x, global_step)
        return x

def resnet18(**kwargs):
    return ResNet(layers=[2,2,2,2],block=BasicBlock,**kwargs)

def resnet34(**kwargs):
    return ResNet(layers=[3,4,6,3], block=BasicBlock, **kwargs)