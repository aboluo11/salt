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


class FinalConv(nn.Module):
    def __init__(self, final_c, writer):
        super().__init__()
        self.conv1 = ConvBlock(final_c, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 1, kernel_size=1)
        self.writer = writer

    def forward(self, x, global_step=None):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class HasSalt(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.linear1 = nn.Linear(in_c, 1)
        # self.linear2 = nn.Linear(256, 1)
        # self.bn = nn.BatchNorm1d(256)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.linear1(x)
        # x = self.bn(torch.relu(x))
        # x = self.linear2(x)
        x = x.view(-1)
        return x

class UnetBlock(nn.Module):
    def __init__(self, feature_c, x_c, out_c, feature_width, drop, writer, layer_num):
        """input channel size: feature_c, x_c
        output channel size: out_c
        """
        super().__init__()
        self.feature_width = feature_width
        # self.gcn = GCN(feature_c, feature_c, 7)
        self.upconv = nn.ConvTranspose2d(x_c, feature_c, kernel_size=3, stride=2, padding=1)
        # self.br1 = BR(out_c)
        # self.br2 = BR(out_c)
        # if self.feature_width != 101:
        #     self.ob_context = ObjectContext(in_c=feature_c, key_c=feature_c//2, value_c=feature_c//2, out_c=feature_c)
        # self.sc = SCBlock(out_c)
        self.conv1 = ConvBlock(feature_c*2, feature_c, kernel_size=3, stride=1, padding=1)
        self.conv2 = ConvBlock(feature_c, out_c, kernel_size=3, stride=1, padding=1)
        self.writer = writer
        self.layer_num = layer_num
        self.tag = f'decode_layer{layer_num}'

    def forward(self, feature, x, global_step=None):
        # if self.feature_width != 101:
        #     feature = self.ob_context(feature)
        # feature = self.gcn(feature)
        # feature = self.br1(feature)
        x = self.upconv(x, output_size=feature.shape)
        # out = F.interpolate(out, size=list(feature.shape[-2:]), mode='bilinear', align_corners=False)
        out = self.conv1(torch.cat([x, feature], dim=1))
        out = self.conv2(out)
        # out = self.sc(out)
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
        self.features = []
        self.handles = []
        for m in self.encoder.children():
            handle = m.register_forward_pre_hook(lambda module, input: self.features.append(input[0]))
            self.handles.append(handle)
        self.writer = writer
        self.dummy_forward(T(ds[0][0], cuda=False).unsqueeze(0), drop)

        # self.encoder1.conv.weight.data = resnet.conv1.weight.data.mean(dim=1, keepdim=True)

    def forward(self, x, global_step=None):
        """
        return [mask, has_salt(logit)]
        """
        bs = x.shape[0]
        res = torch.zeros(bs, 1, *(x.shape[-2:]), device='cuda')

        x = self.bn_input(x)
        x = self.encoder(x)

        has_salt = self.has_salt(x)

        has_salt_index = torch.sigmoid(has_salt) > 0.5
        if has_salt_index.any():
            x = x[has_salt_index]
        else:
            return res, has_salt

        hyper_columns = []
        for i, (feature, block) in enumerate(zip(reversed(self.features), self.upmodel)):
            x = block(feature[has_salt_index], x, global_step)
            hyper_columns.append(x)

        for i in range(len(hyper_columns)):
            hyper_columns[i] = F.interpolate(hyper_columns[i], size=101, mode='bilinear', align_corners=False)

        self.features = []
        x = torch.cat(hyper_columns, dim=1)
        x = self.final_conv(x, global_step)

        res[has_salt_index] = x

        return res, has_salt

    def get_layer_groups(self):
        return [self.encoder, [self.upmodel, self.final_conv, self.has_salt]]

    def dummy_forward(self, x, drop):
        with torch.no_grad():
            self.encoder.eval()
            x = self.encoder(x)

            self.has_salt = HasSalt(x.shape[1] * x.shape[2] * x.shape[3])

            upmodel = OrderedDict()
            final_c = 0
            decoder_count = 0
            for i in reversed(range(len(self.features))):
                feature = self.features[i]
                if feature.shape[2] != x.shape[2]:
                    decoder_count += 1
                    block = UnetBlock(feature.shape[1], x.shape[1], 64, feature.shape[-1], drop, self.writer, decoder_count)
                    block.eval()
                    x = block(feature, x)
                    upmodel[f'decoder_layer{decoder_count}'] =block
                    final_c += x.shape[1]
                else:
                    self.handles[i].remove()
            self.features = []
            self.upmodel = nn.Sequential(upmodel)
            self.final_conv = FinalConv(final_c, self.writer)
