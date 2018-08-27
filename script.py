from lightai.learner import *
from lightai.dataloader import *
import torchvision.models as models
from functional import *
from model import *
from transform import *
from dataset import *
from metric import *
import torchvision.transforms as transforms

bs = 32
wd = 1e-6
lr = 0.3

def get_data(trn_tsfm):
    trn_tsfm = MyCompose(trn_tsfm,to_np)
    val_tsfm = to_np
    sp_trn_ds = ImageDataset('inputs/gray/sample/train',tsfm=trn_tsfm)
    sp_val_ds = ImageDataset('inputs/gray/sample/val',tsfm=val_tsfm)
    sp_trn_sampler = BatchSampler(sp_trn_ds, bs)
    sp_val_sampler = BatchSampler(sp_val_ds, bs)
    sp_trn_dl = DataLoader(sp_trn_sampler)
    sp_val_dl = DataLoader(sp_val_sampler)
    return sp_trn_ds,sp_trn_dl,sp_val_dl

def reset_learner(trn_tsfm):
    sp_trn_ds,sp_trn_dl,sp_val_dl = get_data(trn_tsfm)
    resnet = resnet18(False)
    model = Dynamic(model_cut(resnet,-2), sp_trn_ds).cuda()
    layer_opt = LayerOptimizer(model)
    learner = Learner(sp_trn_dl, sp_val_dl, model, nn.BCEWithLogitsLoss(),layer_opt,
                    metric=Score, small_better=False, sv_best_path='./model/best')
    return learner

def test(tsfm, epoch):
    print(f'epoch={epoch}, tsfm=\n{tsfm}')
    learner = reset_learner(tsfm)
    learner.fit(n_epochs=epoch, lrs=lr,wds=wd,clr_params=[50,5,0.01,0], print_stats=False)
    print('----------------------------')

distort = Distort(5,5,5)
intensity = MyColorJitter(brightness=0.05)
crop = CropRandom(0.7)
param = {'degrees':0,'resample':Image.BICUBIC}
zoom = MyRandomAffine(scale=[0.5,2],**param)
shift = MyRandomAffine(translate=[0.5,0.5],**param)
shear = MyRandomAffine(shear=45,**param)

tsfm = MyRandomApply([sample_hflip,shift,intensity,MyRandomChoice(
    [distort,zoom,crop,shear],ps=[0.45,0.225,0.225,0.1])],ps=[0.5,0.5,0.5,0.5])

for epoch in [64,128,256]:
    test(tsfm, epoch)
