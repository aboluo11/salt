from lightai.imps import *

class ImageDataset:
    def __init__(self, path, tsfm=None):
        path = Path(path)
        img_path = path/'images'
        mask_path = path/'masks'
        self.img = list(img_path.iterdir())
        self.mask = list(mask_path.iterdir())
        self.tsfm = tsfm
    def __getitem__(self, idx):
        img = Image.open(self.img[idx])
        mask = Image.open(self.mask[idx]).convert('L')
        sample = [img,mask]
        if self.tsfm:
            sample = self.tsfm(sample)
        return sample
    def __len__(self):
        return len(self.img)

class TotalDataset(ImageDataset):
    def __init__(self):
        img_trn_path = Path('inputs/train/images')
        img_val_path = Path('inputs/val/images')
        mask_trn_path = Path('inputs/train/masks')
        mask_val_path = Path('inputs/val/masks')
        self.img = list(img_trn_path.iterdir())+list(img_val_path.iterdir())
        self.mask = list(mask_trn_path.iterdir())+list(mask_val_path.iterdir())