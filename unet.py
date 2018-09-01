from lightai.imps import *
from functional import *

class UnetBlock(nn.Module):
    def __init__(self,feature_c, x_c, drop):
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
        self.drop = nn.Dropout2d(drop)
        
    def forward(self, feature, x):
        out = self.upconv1(x,output_size=feature.shape)
        out = self.bn1(torch.relu(torch.cat([out, feature], dim=1)))
        out = self.drop(out)
        out = self.bn2(torch.relu(self.conv1(out))) + self.bn3(torch.relu(self.upconv2(x,output_size=feature.shape)))
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
        hook_fn = lambda module,input:self.features.append(input[0])
        for m in leaves(encoder):
            handle = m.register_forward_pre_hook(hook_fn)
            self.handles.append(handle)
        self.dummy_forward(T(ds[0][0], cuda=False).unsqueeze(0), drop)

    def forward(self,x):
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

        for feature,block in zip(reversed(self.features),self.upmodel):
            x = block(feature,x)
        self.features = []
        x = self.final_conv(x)
        return x, has_salt
        
    def get_layer_groups(self):
        return [[self.encoder],[self.upmodel,self.final_conv]]

    def dummy_forward(self,x,drop):
        with torch.no_grad():
            self.encoder.eval()
            x = self.encoder(x)
        self.linear1 = nn.Linear(x.shape[1]*x.shape[2]*x.shape[3], 256)
        self.linear2 = nn.Linear(256, 1)
        upmodel = []
        for i in reversed(range(len(self.features))):
            feature = self.features[i]
            if feature.shape[2] != x.shape[2]:
                block = UnetBlock(feature.shape[1], x.shape[1], drop)
                x = block(feature,x)
                upmodel.append(block)
            else:
                self.handles[i].remove()
        self.features = []
        self.upmodel = nn.ModuleList(upmodel)
        self.final_conv = nn.Conv2d(x.shape[1],1,kernel_size=1)