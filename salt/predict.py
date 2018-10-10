from lightai.data import DataLoader
from lightai.core import *

from salt.dataset import TestDataset
from salt.transform import *


def tta_mean_predict(predicts: List, reverse_tta: List):
    """
    :param predicts: list of predict, predict: [logit, has_salt].
    :param reverse_tta: apply to logit
    :return: mean probability of tta predicts, shape: [batch_size]
    """
    assert len(predicts) == len(reverse_tta)
    p_masks = []
    for predict, f in zip(predicts, reverse_tta):
        logit = predict[0]
        if f:
            logit = f(logit)
        p_masks.append(torch.sigmoid(logit))
    return torch.stack(p_masks).mean(dim=0)


def predict_test(models: List, tta_tsfms: List = [None, hflip], reverse_tta: List = [None, hflip], bs = 48):
    """
    :param reverse_tta: apply to predicted mask
    """
    submit = pd.read_csv('inputs/sample_submission.csv')
    test_dl = get_test_data(tta_tsfms, bs)
    for model in models:
        model.eval()
    with torch.no_grad():
        for tta_batch in test_dl:
            model_predicts = []
            for model in models:
                predicts = []
                for img, name in tta_batch:
                    predict = model(img)
                    predicts.append(predict)
                predict = tta_mean_predict(predicts, reverse_tta)
                model_predicts.append(predict)
            ensemble_predict = torch.stack(model_predicts).mean(dim=0) > 0.5
            ensemble_predict = ensemble_predict.cpu().numpy()
            for n, m in zip(name, ensemble_predict):
                m = rl_enc(m)
                submit.loc[submit['id'] == n, 'rle_mask'] = m
    submit.to_csv('submit.csv', index=False)


def rl_enc(img):
    pixels = img.flatten('F')
    a = np.concatenate([[0], pixels])
    b = np.concatenate([pixels, [0]])
    start = np.where(b - a == 1)[0] + 1
    end = np.where(a - b == 1)[0] + 1
    length = end - start
    res = np.zeros(2 * len(start), dtype=int)
    res[::2] = start
    res[1::2] = length
    return ' '.join(str(x) for x in res)


def get_test_data(tta_tsfms: List, bs):
    def tsfm(img):
        img = np.asarray(img).astype(np.float32) / 255
        img = np.expand_dims(img, 0)
        # img = add_depth(img)
        return img

    test_ds = TestDataset(tsfm=tsfm, tta_tsfms=tta_tsfms)
    test_sampler = BatchSampler(test_ds, bs)
    test_dl = DataLoader(test_sampler)
    return test_dl
