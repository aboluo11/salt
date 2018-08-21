from lightai.imps import *

class UnetBlock(nn.Module):
    def __init__(self,feature_c, x_c):
        """input channel size: feature_c, x_c
        output channel size: feature_c
        """
        super().__init__()
        self.upconv = nn.ConvTranspose2d(x_c, feature_c, kernel_size=3, stride=2, padding=1)
        self.conv1 = nn.Conv2d(feature_c*2, feature_c, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(feature_c, feature_c, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(feature_c)
        
    def forward(self, feature, x):
        x = self.upconv(x,output_size=feature.shape)
        x = torch.cat([x, feature], dim=1)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        return self.bn(x)

class Dynamic(nn.Module):
    def __init__(self, encoder, ds):
        super().__init__()
        self.encoder = encoder
        self.features = []
        self.handles = []
        hook_fn = lambda module,input:self.features.append(input[0])
        for m in encoder.children():
            handle = m.register_forward_pre_hook(hook_fn)
            self.handles.append(handle)
        self.dummy_forward(T(np.transpose(ds[0][0],axes=[2,0,1]), cuda=False).unsqueeze(0))

    def forward(self,x):
        x = self.encoder(x)
        for feature,block in zip(reversed(self.features),self.upmodel):
            x = block(feature,x)
        self.features = []
        x = self.final_conv(x)
        return x
        
    def get_layer_groups(self):
        return [self.encoder,[self.upmodel,self.final_conv]]

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
    