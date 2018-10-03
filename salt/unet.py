from lightai.imps import *
from tensorboardX import SummaryWriter
from .log import *
from .resnet import *
from .block import *
from .object_context import *
from .gcn import *


def _leaves(model):
    res = []
    childs = children(model)
    if len(childs) == 0:
        return [model]
    for key, module in model._modules.items():
        if key == 'downsample' or key == 'relu' or key[-4:] == 'gate':
            continue
        res += _leaves(module)
    return res

def _mul(shape):
    res = 1
    for each in shape:
        res *= each
    return res

def _percent(x):
    a = torch.sum(x>0).float()
    b = _mul(x.shape)
    return (a/b).item()


class Fuse(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.conv1 = ConvBlock(in_c, in_c//2, kernel_size=1)
        self.conv2 = nn.Conv2d(in_c//2, 1, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class LogitPixel(nn.Module):
    def __init__(self, in_c, writer):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, 1, kernel_size=1)
        self.writer = writer

    def forward(self, x, global_step=None):
        x = self.conv1(x)
        return x

class LogitImg(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.linear1 = nn.Linear(in_c, 1)

    def forward(self, x):
        x = self.linear1(x)
        x = x.view(-1)
        return x


class FuseImg(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(in_c, out_c)
        self.bn = nn.BatchNorm1d(out_c)

    def forward(self, x):
        bs = x.shape[0]
        x = self.avg_pool(x)
        x = x.view(bs, -1)
        x = self.linear(x)
        x = self.bn(torch.relu(x))
        return x


class UnetBlock(nn.Module):
    def __init__(self, feature_c, x_c, out_c, feature_width, drop, writer, layer_num):
        """input channel size: feature_c, x_c
        output channel size: out_c
        """
        super().__init__()
        self.feature_width = feature_width
        self.upconv = nn.ConvTranspose2d(x_c, x_c, kernel_size=3, stride=2, padding=1)
        self.conv1 = ConvBlock(feature_c + x_c, feature_c, kernel_size=3, stride=1, padding=1)
        self.conv2 = ConvBlock(feature_c, out_c, kernel_size=3, stride=1, padding=1)
        self.writer = writer
        self.layer_num = layer_num
        self.tag = f'decode_layer{layer_num}'
        # self.spatial = SpatialGate(out_c, writer, self.tag)
        # self.ob_context = ObjectContext(feature_c, feature_c//2, feature_c//2, feature_c)

    def forward(self, feature, x, global_step=None):
        x = self.upconv(x, output_size=feature.shape)
        out = self.conv1(torch.cat([x, feature], dim=1))

        # if self.feature_width != 101:
        #     out = self.ob_context(out)

        out = self.conv2(out)
        # out = self.spatial(out, global_step)
        return out


class Dynamic(nn.Module):
    def __init__(self, resnet, ds, drop, linear_drop, writer=None):
        super().__init__()
        self.bn_input = nn.BatchNorm2d(1)
        resnet = resnet(pretrained=True)
        self.encoder1 = ConvBlock(1, 64, 7, stride=1, padding=3)
        self.encoder2 = resnet.layer1
        self.encoder3 = resnet.layer2
        self.encoder4 = resnet.layer3
        self.encoder5 = resnet.layer4
        self.encoder = nn.Sequential(self.encoder1, self.encoder2, self.encoder3, self.encoder4, self.encoder5)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.features = []
        self.handles = []
        for m in self.encoder.children():
            handle = m.register_forward_pre_hook(lambda module, input: self.features.append(input[0]))
            self.handles.append(handle)
        self.writer = writer
        self.dummy_forward(T(ds[0][0]).unsqueeze(0), drop)

        self.encoder1.conv.weight.data = resnet.conv1.weight.data.mean(dim=1, keepdim=True)

    def forward(self, x, global_step=None):
        bs = x.shape[0]
        res = torch.zeros(bs, 1, *(x.shape[-2:]), device='cuda')

        x = self.bn_input(x)
        x = self.encoder(x)

        logit_img = self.logit_img(self.avg_pool(x))
        logit_img = logit_img.view(bs, 1, 1, 1)

        x = self.center(x)

        hyper_columns = []
        for i, (feature, block) in enumerate(zip(reversed(self.features), self.upmodel)):
            x = block(feature, x, global_step)
            hyper_columns.append(x)

        for i in range(len(hyper_columns)):
            hyper_columns[i] = F.interpolate(hyper_columns[i], size=101, mode='bilinear', align_corners=False)

        self.features = []

        x = torch.cat(hyper_columns, dim=1)
        x *= torch.sigmoid(logit_img)
        logit_pixel = self.logit_pixel(x)

        return logit_pixel

    def get_layer_groups(self):
        return [self.encoder, [self.upmodel]]

    def dummy_forward(self, x, drop):
        with torch.no_grad():
            self.encoder.cuda()
            self.encoder.eval()
            x = self.encoder(x)

            self.logit_img = LogitImg(x.shape[1])

            self.center = nn.Sequential(
                ConvBlock(x.shape[1], x.shape[1], kernel_size=3, padding=1),
                ConvBlock(x.shape[1], x.shape[1]//2, kernel_size=3, padding=1)
            ).cuda()
            x = self.center(x)

            upmodel = OrderedDict()
            pixel_c = 0
            decoder_count = 0
            for i in reversed(range(len(self.features))):
                feature = self.features[i]
                if feature.shape[2] != x.shape[2]:
                    decoder_count += 1
                    block = UnetBlock(feature.shape[1], x.shape[1], 64, feature.shape[-1], drop, self.writer, decoder_count)
                    block.cuda()
                    block.eval()
                    x = block(feature, x)
                    upmodel[f'decoder_layer{decoder_count}'] =block
                    pixel_c += x.shape[1]
                else:
                    self.handles[i].remove()
            self.features = []
            self.upmodel = nn.Sequential(upmodel)
            self.logit_pixel = LogitPixel(pixel_c, writer=self.writer)
