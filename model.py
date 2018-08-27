from lightai.imps import *
from functional import *

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
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=1, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(inplace=True)
        x = self.bn1(x)
        x = self.conv2(x)
        x = F.relu(inplace=True)
        x = self.bn2(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        state_dict = model_zoo.load_url(model_urls['resnet18'])
        state_dict['conv1.weight'] = state_dict['conv1.weight'].mean(1,keepdim=True)
        model.load_state_dict(state_dict)
    return model

class UnetBlock(nn.Module):
    def __init__(self,feature_c, x_c):
        """input channel size: feature_c, x_c
        output channel size: feature_c
        """
        super().__init__()
        self.upconv1 = nn.ConvTranspose2d(x_c, feature_c, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv1 = nn.Conv2d(feature_c*2, feature_c, kernel_size=3, stride=1, padding=1, bias=False)
        self.upconv2 = nn.ConvTranspose2d(x_c, feature_c, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(2*feature_c)
        self.bn2 = nn.BatchNorm2d(feature_c)
        self.bn3 = nn.BatchNorm2d(feature_c)
        
    def forward(self, feature, x):
        out = self.upconv1(x,output_size=feature.shape)
        out = self.bn1(torch.relu(torch.cat([out, feature], dim=1)))
        out = self.bn2(torch.relu(self.conv1(out))) + self.bn3(torch.relu(self.upconv2(x,output_size=feature.shape)))
        return out

class Dynamic(nn.Module):
    def __init__(self, encoder, ds):
        super().__init__()
        self.bn_input = nn.BatchNorm2d(1)
        self.encoder = encoder
        self.features = []
        self.handles = []
        hook_fn = lambda module,input:self.features.append(input[0])
        for m in leaves(encoder):
            handle = m.register_forward_pre_hook(hook_fn)
            self.handles.append(handle)
        self.dummy_forward(T(ds[0][0], cuda=False).unsqueeze(0))

    def forward(self,x):
        x = self.bn_input(x)
        x = self.encoder(x)
        for feature,block in zip(reversed(self.features),self.upmodel):
            x = block(feature,x)
        self.features = []
        x = self.final_conv(x)
        return x
        
    def get_layer_groups(self):
        return [[self.encoder],[self.upmodel,self.final_conv]]

    def dummy_forward(self,x):
        with torch.no_grad():
            x = self.encoder(x)
        upmodel = []
        for i in reversed(range(len(self.features))):
            feature = self.features[i]
            if feature.shape[2] != x.shape[2]:
                block = UnetBlock(feature.shape[1], x.shape[1])
                x = block(feature,x)
                upmodel.append(block)
            else:
                self.handles[i].remove()
        self.features = []
        self.upmodel = nn.ModuleList(upmodel)
        self.final_conv = nn.Conv2d(x.shape[1],1,kernel_size=1)
    