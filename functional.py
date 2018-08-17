from lightai.imps import *

def create_val(ds):
    trn_idx, valid_idx = split_idx(len(ds.img), 0.8, seed=1)
    trn_path = Path('inputs/train')
    val_path = Path('inputs/val')
    val_path.mkdir()
    (val_path/'images').mkdir()
    (val_path/'masks').mkdir()
    for idx in valid_idx:
        fname = ds.img[idx].parts[-1]
        (trn_path/'images'/f'{fname}').rename(val_path/'images'/f'{fname}')
        (trn_path/'masks'/f'{fname}').rename(val_path/'masks'/f'{fname}')

def rl_enc(img):
    pixels = img.flatten('F')
    a = np.concatenate([[0],pixels])
    b = np.concatenate([pixels,[0]])
    start = np.where(b - a == 1)[0] + 1
    end = np.where(a - b == 1)[0] + 1
    length = end - start
    res = np.zeros(2*len(start),dtype=int)
    res[::2] = start
    res[1::2] = length
    return ' '.join(str(x) for x in res)

def score(predict, target, threshold):
    """return a batch of images' score at threshold t
    """
    predict, target = predict.view(-1,101*101), target.view(-1,101*101)
    metric = torch.arange(0.5,1,0.05, device='cuda')
    intersection = torch.sum((predict>threshold)*(target==1),dim=1)
    union = torch.sum(predict>threshold,dim=1)+torch.sum(target==1,dim=1)-intersection
    percentage = intersection.float()/union.float()
    percentage[torch.isnan(percentage)] = 1
    percentage = percentage.view(-1,1)
    percentage = percentage.expand(-1,10)
    return torch.sum(percentage>metric,dim=1).float()/10

def thres_score(model, val_dl):
    thresholds = np.linspace(0,1,num=100,endpoint=False)
    res = []
    with torch.no_grad():
        model.eval()
        for img, mask in val_dl:
            batch_res = []
            img,mask = T(img),T(mask)
            predict = torch.sigmoid(model(img))
            for t in thresholds:
                batch_res.append(iou(predict, mask, t))
            res.append(torch.stack(batch_res))
    return thresholds,np.array(torch.cat(res,dim=1).mean(dim=1))

def visualize(ds, id):
    _, ax = plt.subplots(1,2)
    img, mask = ds[id]
    print(ds.img[id])
    ax[0].imshow(img[0])
    ax[1].imshow(mask[0])