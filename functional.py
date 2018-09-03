from lightai.imps import *

def fold_csv_to_trn_val(csv, path, n_fold):
    path = Path(path)
    for k in range(n_fold):
        fold_path = path/f'{k}'
        val_fold = csv.loc[csv['fold']==k]
        trn_fold = csv.loc[csv['fold']!=k]
        val_fold.drop('fold', axis=1, inplace=True)
        trn_fold.drop('fold', axis=1, inplace=True)
        fold_path.mkdir(exist_ok=True)
        val_fold.to_csv(fold_path/'val.csv', index=False)
        trn_fold.to_csv(fold_path/'trn.csv', index=False)

def create_kfold_csv(n_fold=5):
    img_paths = chain(Path('inputs/train/images').iterdir(), Path('inputs/val/images').iterdir())
    img_names = [p.parts[-1].split('.')[0] for p in img_paths]
    depth = pd.read_csv('inputs/depths.csv')
    depth = depth.loc[depth['id'].isin(img_names)]
    depth.sort_values('z', inplace=True)
    depth.drop('z', axis=1, inplace=True)
    depth['fold'] = (list(range(n_fold))*depth.shape[0])[:depth.shape[0]]
    fold_csv_to_trn_val(depth, f'inputs/all', n_fold)

def create_sample_kfold_csv(n_fold=5):
    sample = pd.read_csv('inputs/all/0/val.csv')
    sample['fold'] = (list(range(n_fold))*sample.shape[0])[:sample.shape[0]]
    fold_csv_to_trn_val(sample, f'inputs/sample', n_fold)

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

def thres_score(models, val_dls, reverse_tta):
    """res: shape: [thresholds], item: score
    """
    thresholds = np.linspace(0,1,num=100,endpoint=False)
    model_res = []
    res = []
    with torch.no_grad():
        for model, val_dl in zip(models, val_dls):
            model.eval()
            for batch in val_dl:
                batch_res = []
                predicts = []
                assert len(batch) == len(reverse_tta)
                for [img, mask], f in zip(batch,reverse_tta):
                    img,mask = T(img),T(mask)
                    predict, has_salt = model(img)
                    predict, has_salt = torch.sigmoid(predict), torch.sigmoid(has_salt)
                    if (has_salt<0.5).sum():
                        predict[has_salt < 0.5] = torch.zeros_like(predict[0], dtype=torch.float32, device='cuda')
                    if f:
                        predict = f(predict)
                    predicts.append(predict)
                predict = torch.stack(predicts).mean(dim=0)
                for t in thresholds:
                    batch_res.append(score(predict, mask, t))
                model_res.append(torch.stack(batch_res))
            res.append(torch.cat(model_res,dim=1).mean(dim=1))
    return thresholds, np.array(torch.stack(res).mean(dim=0))

def cal_mean_std(trn_dl):
    total = []
    for img, mask in trn_dl:
        total.append(img)
    total = np.concatenate(total)
    mean = total.mean((0,2,3))
    std = total.std((0,2,3))
    return mean,std

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

def addindent(s_):
    s = s_.split('\n')
    s = ['  ' + line for line in s]
    s = '\n'.join(s)
    return s