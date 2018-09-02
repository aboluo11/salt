from lightai.imps import *

class CsvDataset:
    def __init__(self, csv_path, tsfm=None, tta_tsfms=None):
        """
        tta_tsfms: list
        tta_tsfms is None: return sample
        tta_tsfm non-empty list: return samples
        """
        self.names = pd.read_csv(csv_path)
        self.img_path = Path('inputs/gray/train/images')
        self.mask_path = Path('inputs/gray/train/masks')
        self.tsfm = tsfm
        self.tta_tsfms = tta_tsfms

    def __getitem__(self, idx):
        name = self.names.iloc[idx].item() + '.png'
        img = Image.open(self.img_path/name)
        mask = Image.open(self.mask_path/name)
        sample = [img, mask]
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
        return len(self.names)

class ImageDataset:
    def __init__(self, path, tsfm=None, tta_tsfms=None):
        """
        tta_tsfms: list
        tta_tsfms is None: return sample
        tta_tsfm non-empty list: return samples
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

class TestDataset():
    def __init__(self,tsfm=None,tta_tsfms=None):
        img_path = Path('inputs/gray/test/images')
        self.img = list(img_path.iterdir())
        self.tsfm = tsfm
        self.tta_tsfms = tta_tsfms
        
    def __getitem__(self, idx):
        img = Image.open(self.img[idx])
        imgs = []
        if self.tta_tsfms:
            for t in self.tta_tsfms:
                if t:
                    imgs.append(t(img))
                else:
                    imgs.append(img)
        if self.tsfm:
            for i, img in enumerate(imgs):
                imgs[i] = self.tsfm(img)
        name = self.img[idx].parts[-1].split('.')[0]
        imgs = [[img,name] for img in imgs]
        return imgs
    
    def __len__(self):
        return len(self.img)
