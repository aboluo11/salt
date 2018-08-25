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
p = 0.5

def get_data(*trn_tsfm):
    trn_tsfm = MyRandomChoice([Hflip(),*trn_tsfm],0.75)
    # trn_tsfm = MyRandomApply([Hflip(),*trn_tsfm],p)
    trn_tsfm = compose(trn_tsfm,to_np)
    val_tsfm = compose(to_np)
    sp_trn_ds = ImageDataset('inputs/gray/sample/train',tsfm=trn_tsfm)
    sp_val_ds = ImageDataset('inputs/gray/sample/val',tsfm=val_tsfm)
    sp_trn_sampler = BatchSampler(sp_trn_ds, bs)
    sp_val_sampler = BatchSampler(sp_val_ds, bs)
    sp_trn_dl = DataLoader(sp_trn_sampler)
    sp_val_dl = DataLoader(sp_val_sampler)
    return sp_trn_ds,sp_trn_dl,sp_val_dl

def reset_learner(*trn_tsfm):
    sp_trn_ds,sp_trn_dl,sp_val_dl = get_data(*trn_tsfm)
    resnet = resnet18(False)
    model = Dynamic(model_cut(resnet,-2), sp_trn_ds).cuda()
    layer_opt = LayerOptimizer(model)
    learner = Learner(sp_trn_dl, sp_val_dl, model, nn.BCEWithLogitsLoss(),layer_opt,
                    metric=Score, small_better=False)
    return learner

def test(*tsfm):
    learner = reset_learner(*tsfm)
    learner.fit(n_epochs=128, lrs=lr,wds=wd,clr_params=[50,5,0.01,0], print_stats=False)
    print('----------------------------')

distort = Distort(5,5,10)
intensity = apply_to_img(transforms.ColorJitter(brightness=0.5))
param = {'degrees':0,'resample':Image.BICUBIC}
zoom = MyRandomAffine(scale=[0.5,2],**param)
shift = MyRandomAffine(translate=[0.5,0.5],**param)
shear = MyRandomAffine(shear=45,**param)

print('random choice')
test(distort,intensity,zoom,shift,shear)