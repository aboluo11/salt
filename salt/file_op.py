from lightai.core import *


def fold_csv_to_trn_val(csv, path, n_fold):
    path = Path(path)
    path.mkdir(exist_ok=True)
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
    train = pd.read_csv('inputs/train.csv')
    salt_coverage = pd.DataFrame(train['id'].copy())
    salt_coverage['id'] = salt_coverage['id'] + '.png'
    salt_coverage = sort_by_coverage(train, salt_coverage)
    salt_coverage.drop('coverage', axis=1, inplace=True)
    salt_coverage['fold'] = (list(range(n_fold)) * salt_coverage.shape[0])[:salt_coverage.shape[0]]
    fold_csv_to_trn_val(salt_coverage, 'inputs/all', n_fold)

def create_kfold_csv_depth(out_name, n_fold=5):
    train = pd.read_csv('inputs/train.csv')
    depth = pd.read_csv('inputs/depths.csv')
    depth = depth.loc[depth['id'].isin(train['id'])]
    depth['id'] = depth['id'] + '.png'
    depth.sort_values('z', inplace=True)
    depth.drop('z', axis=1, inplace=True)
    depth['fold'] = (list(range(n_fold)) * depth.shape[0])[:depth.shape[0]]
    fold_csv_to_trn_val(depth, f'inputs/{out_name}', n_fold)

def create_sample_kfold_csv(in_name, out_name, n_fold=5):
    sample = pd.read_csv(f'inputs/{in_name}/0/val.csv')
    sample['fold'] = (list(range(n_fold)) * sample.shape[0])[:sample.shape[0]]
    fold_csv_to_trn_val(sample, f'inputs/{out_name}', n_fold)

def cal_salt_coverage(mask):
    mask_pixels = mask.sum()
    coverage = mask_pixels/(101*101)
    return coverage

def sort_by_coverage(train, salt_coverage):
    coverage = np.random.randn(len(salt_coverage))
    for i, name in enumerate(train['id']):
        path = f'inputs/gray/train/masks/{name}.png'
        img = Image.open(path)
        img = np.asarray(img).astype(np.float32)/255
        coverage[i] = cal_salt_coverage(img)
    salt_coverage = salt_coverage.assign(coverage=coverage)
    salt_coverage.sort_values('coverage', inplace=True)
    return salt_coverage

def cal_mean_std(trn_dl):
    total = []
    for img, mask in trn_dl:
        total.append(img)
    total = np.concatenate(total)
    mean = total.mean((0, 2, 3))
    std = total.std((0, 2, 3))
    return mean, std