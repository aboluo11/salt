from lightai.imps import *
from tensorboardX import SummaryWriter


def _leaves(model):
    res = []
    childs = children(model)
    if len(childs) == 0:
        return [model]
    for key, module in model._modules.items():
        if key == 'downsample' or key == 'relu':
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
        self.conv1 = nn.Conv2d(final_c, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 1, kernel_size=1)
        self.bn = nn.BatchNorm2d(64)
        self.writer = writer

    def forward(self, x, global_step=None):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.bn(x)
        x = self.conv2(x)
        return x

class UnetBlock(nn.Module):
    def __init__(self, feature_c, x_c, out_c, drop, writer, layer_num):
        """input channel size: feature_c, x_c
        output channel size: out_c
        """
        super().__init__()
        self.upconv1 = nn.ConvTranspose2d(x_c, x_c, kernel_size=3, stride=2, padding=1)
        self.conv1 = nn.Conv2d(feature_c + x_c, out_c, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(feature_c + x_c)
        self.bn2 = nn.BatchNorm2d(out_c)
        # self.drop = nn.Dropout2d(drop)
        self.writer = writer
        self.layer_num = layer_num

    def forward(self, feature, x, global_step=None):
        out = self.upconv1(x, output_size=feature.shape)
        # out = F.interpolate(x, size=list(feature.shape[-2:]), mode='bilinear', align_corners=False)
        # feature = self.bn1(torch.relu(feature))
        out = self.bn1(torch.relu(torch.cat([out, feature], dim=1)))
        # out = self.drop(out)
        out = self.conv1(out)
        out = self.bn2(torch.relu(out))

        if self.writer and global_step:
            self.writer.add_scalar(f'decode_layer{self.layer_num}_grad_mean', self.conv1.weight.grad.mean(), global_step)
            self.writer.add_scalar(f'decode_layer{self.layer_num}_grad_std', self.conv1.weight.grad.std(), global_step)

        return out


class Dynamic(nn.Module):
    def __init__(self, encoder, ds, drop, linear_drop, writer=None):
        super().__init__()
        self.bn_input = nn.BatchNorm2d(1)
        self.encoder = encoder
        self.linear_drop1 = nn.Dropout(linear_drop)
        self.linear_drop2 = nn.Dropout(linear_drop)
        self.linear_bn = nn.BatchNorm1d(256)
        self.features = []
        self.handles = []
        for m in _leaves(encoder):
            handle = m.register_forward_pre_hook(lambda module, input: self.features.append(input[0]))
            self.handles.append(handle)
        self.writer = writer
        self.dummy_forward(T(ds[0][0], cuda=False).unsqueeze(0), drop)

    def forward(self, x, global_step=None):
        """
        return [mask, has_salt(logit)]
        """
        x = self.bn_input(x)
        x = self.encoder(x, global_step)

        has_salt = x.view(x.shape[0], -1)
        has_salt = self.linear_drop1(has_salt)
        has_salt = self.linear1(has_salt)
        has_salt = torch.relu(has_salt)
        has_salt = self.linear_bn(has_salt)
        has_salt = self.linear_drop2(has_salt)
        has_salt = self.linear2(has_salt).view(-1)

        hyper_columns = []
        for i, (feature, block) in enumerate(zip(reversed(self.features), self.upmodel)):
            x = block(feature, x, global_step)
            if x.shape[-1] != 101:
                hyper_columns.append(F.interpolate(x, size=101, mode='bilinear', align_corners=False))
            else:
                hyper_columns.append(x)
        self.features = []
        x = torch.cat(hyper_columns, dim=1)
        x = self.final_conv(x, global_step)
        return x, has_salt

    def get_layer_groups(self):
        return [[self.encoder], [self.upmodel, self.final_conv]]

    def dummy_forward(self, x, drop):
        with torch.no_grad():
            self.encoder.eval()
            x = self.encoder(x, None)
        self.linear1 = nn.Linear(x.shape[1] * x.shape[2] * x.shape[3], 256)
        self.linear2 = nn.Linear(256, 1)
        upmodel = OrderedDict()
        final_c = 0
        decoder_count = 0
        for i in reversed(range(len(self.features))):
            feature = self.features[i]
            if feature.shape[2] != x.shape[2]:
                decoder_count += 1
                block = UnetBlock(feature.shape[1], x.shape[1], 64, drop, self.writer, decoder_count)
                x = block(feature, x)
                upmodel[f'decoder_layer{decoder_count}'] = block
                final_c += x.shape[1]
            else:
                self.handles[i].remove()
        self.features = []
        self.upmodel = nn.Sequential(upmodel)

        # final_c = x.shape[1]

        self.final_conv = FinalConv(final_c, self.writer)
