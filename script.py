from lightai.learner import *
from lightai.dataloader import *
import torchvision.models as models
from functional import *
from model import *
from transform import *
from dataset import *
from metric import *
import torchvision.transforms as transforms
from wide_res import *

bs = 16
wd = 1e-6
lr = 0.3

def get_data(trn_tsfm):
    trn_tsfm = MyCompose(trn_tsfm,to_np)
    val_tsfm = to_np
    sp_trn_ds = ImageDataset('inputs/gray/sample/train',tsfm=trn_tsfm)
    sp_val_ds = ImageDataset('inputs/gray/sample/val',tsfm=val_tsfm)
    sp_trn_sampler = BatchSampler(sp_trn_ds, bs)
    sp_val_sampler = BatchSampler(sp_val_ds, int(bs*1.5))
    sp_trn_dl = DataLoader(sp_trn_sampler)
    sp_val_dl = DataLoader(sp_val_sampler)
    return sp_trn_ds,sp_trn_dl,sp_val_dl

def reset_learner(trn_tsfm,width,drop):
    sp_trn_ds,sp_trn_dl,sp_val_dl = get_data(trn_tsfm)
    # resnet = resnet18(False, width=width,drop=drop)
    resnet = Wide_ResNet(16, width, 0.3)
    model = Dynamic(resnet, sp_trn_ds, drop=drop).cuda()
    layer_opt = LayerOptimizer(model)
    learner = Learner(sp_trn_dl, sp_val_dl, model, nn.BCEWithLogitsLoss(),layer_opt,
                    metric=Score, small_better=False, sv_best_path='./model/best')
    return learner

def test(tsfm, epoch, width, drop):
    print((
        f'epoch={epoch}\n'
        f'width={width}\n'
        f'drop={drop}\n'
        f'bs={bs}\n'
        f'tsfm={tsfm}'
    ))
    learner = reset_learner(tsfm,width,drop)
    learner.fit(n_epochs=epoch, lrs=lr,wds=wd,clr_params=[50,5,0.01,0.1], print_stats=False)
    print('----------------------------')

distort = Distort(5,5,5)
intensity = MyColorJitter(brightness=0.05)
crop = CropRandom(0.7)
param = {'degrees':0,'resample':Image.BICUBIC}
zoom = MyRandomAffine(scale=[0.5,2],**param)
zoom_in = MyRandomAffine(scale=[1,2],**param)
zoom_out = MyRandomAffine(scale=[0.5,1],**param)
shift = MyRandomAffine(translate=[0.5,0.5],**param)
shear = MyRandomAffine(shear=45,**param)

tsfm1 = MyRandomApply([sample_hflip,shift,intensity,MyRandomChoice(
    [distort,zoom,crop,shear],ps=[0.45,0.225,0.225,0.1])],ps=[0.5,0.5,0.5,0.5])

tsfm2 = MyRandomApply([sample_hflip,shift,MyRandomChoice(
    [distort,zoom,crop,shear],ps=[0.45,0.225,0.225,0.1])],ps=[0.5,0.5,0.5])

tsfm3 = MyRandomApply([sample_hflip,shift,MyRandomChoice(
    [Distort(5,5,10),zoom,crop,shear],ps=[0.45,0.225,0.225,0.1])],ps=[0.5,0.5,0.5])

for epoch in [32]:
    test(tsfm1,epoch,width=4,drop=0)
