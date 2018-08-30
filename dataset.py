from lightai.imps import *

class ImageDataset:
    def __init__(self, path, tsfm=None, tta_tsfms=None):
        """
        tta_tsfms: list
        train time: tta_tsfms is None, return sample
        test time: tta_tsfm must be non-empty list, return samples
        """
        path = Path(path)
        img_path = path/'images'
        mask_path = path/'masks'
        self.img = list(img_path.iterdir())
        self.mask = list(mask_path.iterdir())
        self.tsfm = tsfm
        self.tta_tsfms = tta_tsfms
    def __getitem__(self, idx):
        img = Image.open(self.img[idx])
        mask = Image.open(self.mask[idx])
        sample = [img,mask]
        if self.tta_tsfms:
            samples = []
            for t in self.tta_tsfms:
                if t:
                    samples.append(t(sample))
                else:
                    samples.append(sample)
            if self.tsfm:
                for i, sample in enumerate(samples):
                    samples[i] = self.tsfm(sample)
            return samples
        else:
            if self.tsfm:
                sample = self.tsfm(sample)
            return sample
    def __len__(self):
        return len(self.img)

class TotalDataset(ImageDataset):
    def __init__(self):
        img_trn_path = Path('inputs/gray/train/images')
        img_val_path = Path('inputs/gray/val/images')
        mask_trn_path = Path('inputs/gray/train/masks')
        mask_val_path = Path('inputs/gray/val/masks')
        self.img = list(img_trn_path.iterdir())+list(img_val_path.iterdir())
        self.mask = list(mask_trn_path.iterdir())+list(mask_val_path.iterdir())