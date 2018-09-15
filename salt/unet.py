from lightai.imps import *


def leaves(model):
    res = []
    childs = children(model)
    if len(childs) == 0:
        return [model]
    for key, module in model._modules.items():
        if key == 'downsample' or key == 'relu':
            continue
        res += leaves(module)
    return res

class UnetBlock(nn.Module):
    def __init__(self, feature_c, x_c, out_c, drop):
        """input channel size: feature_c, x_c
        output channel size: feature_c
        """
        super().__init__()
        self.upconv1 = nn.ConvTranspose2d(x_c, x_c, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv1 = nn.Conv2d(feature_c + x_c, out_c, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(feature_c + x_c)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.drop = nn.Dropout2d(drop)

    def forward(self, feature, x):
        out = self.upconv1(x, output_size=feature.shape)
        out = self.bn1(torch.relu(torch.cat([out, feature], dim=1)))
        out = self.drop(out)
        out = self.bn2(torch.relu(self.conv1(out)))
        return out


class Dynamic(nn.Module):
    def __init__(self, encoder, ds, drop, linear_drop):
        super().__init__()
        self.bn_input = nn.BatchNorm2d(1)
        self.encoder = encoder
        self.linear_drop1 = nn.Dropout(linear_drop)
        self.linear_drop2 = nn.Dropout(linear_drop)
        self.linear_bn = nn.BatchNorm1d(256)
        self.features = []
        self.handles = []
        for m in leaves(encoder):
            handle = m.register_forward_pre_hook(lambda module, input: self.features.append(input[0]))
            self.handles.append(handle)
        self.dummy_forward(T(ds[0][0], cuda=False).unsqueeze(0), drop)

    def forward(self, x):
        """
        return [mask, has_salt(logit)]
        """
        x = self.bn_input(x)
        x = self.encoder(x)

        has_salt = x.view(x.shape[0], -1)
        has_salt = self.linear_drop1(has_salt)
        has_salt = self.linear1(has_salt)
        has_salt = torch.relu(has_salt)
        has_salt = self.linear_bn(has_salt)
        has_salt = self.linear_drop2(has_salt)
        has_salt = self.linear2(has_salt).view(-1)

        hyper_columns = []
        for feature, block in zip(reversed(self.features), self.upmodel):
            x = block(feature, x)
            if x.shape[-1] != 101:
                hyper_columns.append(F.interpolate(x, size=101, mode='bilinear', align_corners=False))
            else:
                hyper_columns.append(x)
        self.features = []
        x = torch.cat(hyper_columns, dim=1)
        x = self.final_conv(x)
        return x, has_salt

    def get_layer_groups(self):
        return [[self.encoder], [self.upmodel, self.final_conv]]

    def dummy_forward(self, x, drop):
        with torch.no_grad():
            self.encoder.eval()
            x = self.encoder(x)
        self.linear1 = nn.Linear(x.shape[1] * x.shape[2] * x.shape[3], 256)
        self.linear2 = nn.Linear(256, 1)
        upmodel = []
        final_c = 0
        for i in reversed(range(len(self.features))):
            feature = self.features[i]
            if feature.shape[2] != x.shape[2]:
                block = UnetBlock(feature.shape[1], x.shape[1], 64, drop)
                x = block(feature, x)
                upmodel.append(block)
                final_c += x.shape[1]
            else:
                self.handles[i].remove()
        self.features = []
        self.upmodel = nn.ModuleList(upmodel)
        self.final_conv = nn.Sequential(
            nn.Conv2d(final_c, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 1, kernel_size=1)
        )
