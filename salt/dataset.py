from .transform import *


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
        name = self.names.iloc[idx].item()
        img = Image.open(self.img_path / name)
        mask = Image.open(self.mask_path / name)
        sample = [img, mask]
        if self.tta_tsfms:
            samples = []
            for t in self.tta_tsfms:
                if t:
                    samples.append([t(img), mask])
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


class TestDataset:
    """
    always tta
    """
    def __init__(self, tta_tsfms: List, tsfm=None):
        img_path = Path('inputs/gray/test/images')
        self.img_path = list(img_path.iterdir())
        self.tsfm = tsfm
        self.tta_tsfms = tta_tsfms

    def __getitem__(self, idx):
        img = Image.open(self.img_path[idx])
        imgs = []
        for t in self.tta_tsfms:
            if t:
                imgs.append(t(img))
            else:
                imgs.append(img)
        if self.tsfm:
            for i, img in enumerate(imgs):
                imgs[i] = self.tsfm(img)
        name = self.img_path[idx].parts[-1].split('.')[0]
        imgs = [[img, name] for img in imgs]
        return imgs

    def __len__(self):
        return len(self.img_path)



