from lightai.core import *


def fold_csv_to_trn_val(csv, path, n_fold):
    path = Path(path)
    for k in range(n_fold):
        fold_path = path / f'{k}'
        val_fold = csv.loc[csv['fold'] == k]
        trn_fold = csv.loc[csv['fold'] != k]
        val_fold.drop('fold', axis=1, inplace=True)
        trn_fold.drop('fold', axis=1, inplace=True)
        fold_path.mkdir(exist_ok=True)
        val_fold.to_csv(fold_path / 'val.csv', index=False)
        trn_fold.to_csv(fold_path / 'trn.csv', index=False)


def create_kfold_csv(n_fold=5):
    img_paths = Path('inputs/gray/train/images').iterdir()
    img_names = [p.parts[-1].split('.')[0] for p in img_paths]
    depth = pd.read_csv('inputs/depths.csv')
    depth = depth.loc[depth['id'].isin(img_names)]
    depth['id'] = depth['id'] + '.png'
    depth.sort_values('z', inplace=True)
    depth.drop('z', axis=1, inplace=True)
    depth['fold'] = (list(range(n_fold)) * depth.shape[0])[:depth.shape[0]]
    fold_csv_to_trn_val(depth, f'inputs/all', n_fold)


def create_sample_kfold_csv(n_fold=5):
    sample = pd.read_csv('inputs/all/0/val.csv')
    sample['fold'] = (list(range(n_fold)) * sample.shape[0])[:sample.shape[0]]
    fold_csv_to_trn_val(sample, f'inputs/sample', n_fold)


def cal_mean_std(trn_dl):
    total = []
    for img, mask in trn_dl:
        total.append(img)
    total = np.concatenate(total)
    mean = total.mean((0, 2, 3))
    std = total.std((0, 2, 3))
    return mean, std
