from lightai.imps import *
from torchvision.transforms import *
import torchvision.transforms


# mean: 0.47146264
# std: 0.1610698

def addindent(s_):
    s = s_.split('\n')
    s = ['  ' + line for line in s]
    s = '\n'.join(s)
    return s


def to_np(sample):
    img, mask = sample
    img = np.asarray(img).astype(np.float32) / 255
    mask = np.asarray(mask).astype(np.float32) / 255
    img = np.expand_dims(img, 0)
    mask = np.expand_dims(mask, 0)
    return img, mask


def hflip(img):
    return img.transpose(Image.FLIP_LEFT_RIGHT)


def sample_hflip(sample):
    img, mask = sample
    img = img.transpose(Image.FLIP_LEFT_RIGHT)
    mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
    return [img, mask]


def img_hflip(sample):
    img, mask = sample
    img = img.transpose(Image.FLIP_LEFT_RIGHT)
    return [img, mask]


def mask_hflip(mask):
    """mask: pytorch tensor, shape: [bs,...]"""
    mask = mask.flip(dims=[len(mask.shape) - 1])
    return mask


class MyColorJitter(ColorJitter):
    def __call__(self, sample):
        img, mask = sample
        img = super().__call__(img)
        return [img, mask]


class MultiChildTsfm:
    def __repr__(self):
        res = f'{self.__class__.__name__}('
        for t, p in zip(self.tsfms, self.ps):
            child_str = t.__name__ if isinstance(t, types.FunctionType) else repr(t)
            child_str = addindent(child_str)
            res += f'\n{child_str}, p={p}'
        res += f'\n)'
        return res


class MyCompose(MultiChildTsfm):
    def __init__(self, *tsfms):
        self.tsfms = tsfms

    def __call__(self, sample):
        for t in self.tsfms:
            if t:
                sample = t(sample)
        return sample


class MyRandomApply(MultiChildTsfm):
    def __init__(self, tsfms, ps):
        self.tsfms = tsfms
        self.ps = listify(ps, tsfms)

    def __call__(self, sample):
        for t, p in zip(self.tsfms, self.ps):
            if rand() < p:
                sample = t(sample)
        return sample


class MyRandomChoice(MultiChildTsfm):
    def __init__(self, tsfms, ps):
        self.tsfms = tsfms
        self.ps = listify(ps, tsfms)

    def __call__(self, sample):
        t = np.random.choice(self.tsfms, p=self.ps)
        sample = t(sample)
        return sample


class MyRandomAffine(RandomAffine):
    def __call__(self, sample):
        img, mask = sample
        ret = self.get_params(self.degrees, self.translate, self.scale, self.shear, img.size)
        img = torchvision.transforms.functional.affine(img, *ret, resample=self.resample, fillcolor=self.fillcolor)
        mask = torchvision.transforms.functional.affine(mask, *ret, resample=self.resample, fillcolor=self.fillcolor)
        return [img, mask]


class Distort:
    def __init__(self, horizontal_tiles, vertical_tiles, magnitude):
        self.horizontal_tiles = horizontal_tiles
        self.vertical_tiles = vertical_tiles
        self.magnitude = magnitude

    def __call__(self, sample):
        img, mask = sample
        w, h = img.size

        horizontal_tiles = self.horizontal_tiles
        vertical_tiles = self.vertical_tiles

        width_of_square = int(w // float(horizontal_tiles))
        height_of_square = int(h // float(vertical_tiles))

        width_of_last_square = w - (width_of_square * (horizontal_tiles - 1))
        height_of_last_square = h - (height_of_square * (vertical_tiles - 1))

        dimensions = []

        for vertical_tile in range(vertical_tiles):
            for horizontal_tile in range(horizontal_tiles):
                if vertical_tile == (vertical_tiles - 1) and horizontal_tile == (horizontal_tiles - 1):
                    dimensions.append([horizontal_tile * width_of_square,
                                       vertical_tile * height_of_square,
                                       width_of_last_square + (horizontal_tile * width_of_square),
                                       height_of_last_square + (height_of_square * vertical_tile)])
                elif vertical_tile == (vertical_tiles - 1):
                    dimensions.append([horizontal_tile * width_of_square,
                                       vertical_tile * height_of_square,
                                       width_of_square + (horizontal_tile * width_of_square),
                                       height_of_last_square + (height_of_square * vertical_tile)])
                elif horizontal_tile == (horizontal_tiles - 1):
                    dimensions.append([horizontal_tile * width_of_square,
                                       vertical_tile * height_of_square,
                                       width_of_last_square + (horizontal_tile * width_of_square),
                                       height_of_square + (height_of_square * vertical_tile)])
                else:
                    dimensions.append([horizontal_tile * width_of_square,
                                       vertical_tile * height_of_square,
                                       width_of_square + (horizontal_tile * width_of_square),
                                       height_of_square + (height_of_square * vertical_tile)])

        last_column = []
        for i in range(vertical_tiles):
            last_column.append((horizontal_tiles - 1) + horizontal_tiles * i)

        last_row = range((horizontal_tiles * vertical_tiles) - horizontal_tiles, horizontal_tiles * vertical_tiles)

        polygons = []
        for x1, y1, x2, y2 in dimensions:
            polygons.append([x1, y1, x1, y2, x2, y2, x2, y1])

        polygon_indices = []
        for i in range((vertical_tiles * horizontal_tiles) - 1):
            if i not in last_row and i not in last_column:
                polygon_indices.append([i, i + 1, i + horizontal_tiles, i + 1 + horizontal_tiles])

        dxy = []
        for _ in polygon_indices:
            dx = np.random.randint(-self.magnitude, self.magnitude + 1)
            dy = np.random.randint(-self.magnitude, self.magnitude + 1)
            dxy.append([dx, dy])

        def do(image):
            for [a, b, c, d], [dx, dy] in zip(polygon_indices, dxy):
                x1, y1, x2, y2, x3, y3, x4, y4 = polygons[a]
                polygons[a] = [x1, y1,
                               x2, y2,
                               x3 + dx, y3 + dy,
                               x4, y4]

                x1, y1, x2, y2, x3, y3, x4, y4 = polygons[b]
                polygons[b] = [x1, y1,
                               x2 + dx, y2 + dy,
                               x3, y3,
                               x4, y4]

                x1, y1, x2, y2, x3, y3, x4, y4 = polygons[c]
                polygons[c] = [x1, y1,
                               x2, y2,
                               x3, y3,
                               x4 + dx, y4 + dy]

                x1, y1, x2, y2, x3, y3, x4, y4 = polygons[d]
                polygons[d] = [x1 + dx, y1 + dy,
                               x2, y2,
                               x3, y3,
                               x4, y4]

            generated_mesh = []
            for i in range(len(dimensions)):
                generated_mesh.append([dimensions[i], polygons[i]])
            return image.transform(image.size, Image.MESH, generated_mesh, resample=Image.BICUBIC)

        img = do(img)
        mask = do(mask)
        return [img, mask]

    def __repr__(self):
        res = f'{self.__class__.__name__}(' \
              f'horizontal_tiles={self.horizontal_tiles}, ' \
              f'vertical_tiles={self.vertical_tiles}, ' \
              f'magnitude={self.magnitude}' \
              f')'
        return res


class CropRandom:
    def __init__(self, percentage_area):
        self.percentage_area = percentage_area

    def __call__(self, sample):
        img, mask = sample
        w, h = img.size

        w_new = int(math.floor(w * self.percentage_area))
        h_new = int(math.floor(h * self.percentage_area))

        random_left_shift = np.random.randint(0, int(w - w_new))  # Note: randint() is from uniform distribution.
        random_down_shift = np.random.randint(0, int(h - h_new))

        def do(image):
            image = image.crop(
                (random_left_shift, random_down_shift, w_new + random_left_shift, h_new + random_down_shift))
            return image.resize([101, 101], resample=Image.LANCZOS)

        img = do(img)
        mask = do(mask)

        return [img, mask]

    def __repr__(self):
        res = f'{self.__class__.__name__}(' \
              f'percentage_area={self.percentage_area}' \
              f')'
        return res
