from .predict import *


class Score:
    """
    support tta
    """

    def __init__(self, reverse_ttas):
        self.scores = []
        self.reverse_ttas = reverse_ttas

    def __call__(self, predicts, target: torch.Tensor):
        """
        :param predicts: list of predicts or predict if no tta.
            predict: tta [mask, has_salt], all items are logit tensor
        :param target: mask, logit tensor
        calculate a batch of predicts' scores, shape: [batch_size], and add it to self.scores
        """
        p_mask = tta_mean_predict(predicts, self.reverse_ttas)
        s = get_score(p_mask, target)
        self.scores.append(s)

    def res(self) -> float:
        """
        :return:  batches of predicts' score, reduced
        """
        res = torch.cat(self.scores).mean().item()
        self.scores = []
        return res

class NoSaltCorrectRate:
    def __init__(self, reverse_ttas):
        self.reverse_ttas = reverse_ttas
        self.t_no_salt = 0
        self.p_no_salt = 0

    def __call__(self, predicts, target):
        p_logit = tta_mean_predict(predicts, self.reverse_ttas)
        t_no_salt_index = ~(target.byte().view(target.shape[0], -1).any(dim=1))
        if t_no_salt_index.any():
            p_logit = p_logit[t_no_salt_index]
            self.p_no_salt += (~((p_logit>0.5).view(p_logit.shape[0], -1).any(dim=1))).sum()
            self.t_no_salt += t_no_salt_index.sum()

    def res(self):
        res = self.p_no_salt.item()/self.t_no_salt.item()
        self.t_no_salt, self.p_no_salt = 0, 0
        return res

class HasSaltScore:
    def __init__(self, reverse_ttas):
        self.reverse_ttas = reverse_ttas
        self.scores = []

    def __call__(self, predicts, target):
        p_logit = tta_mean_predict(predicts, self.reverse_ttas)
        t_has_salt_index = target.byte().view(target.shape[0], -1).any(dim=1)
        if t_has_salt_index.any():
            s = get_score(p_logit[t_has_salt_index], target[t_has_salt_index])
            self.scores.append(s)

    def res(self):
        res = torch.cat(self.scores).mean().item()
        self.scores = []
        return res

def iou_to_score(iou):
    shape = iou.shape
    iou = iou.view(*shape, 1).expand(*shape, 10)
    metric = torch.arange(0.5, 1, 0.05, device='cuda')
    return torch.sum(iou > metric, dim=-1).float() / 10

def get_score(predict: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    :param predict: batch of predict masks
    :param target: batch of target masks
    :return: a batch of images' score at threshold 0.5, shape: [batch_size]
    """
    predict, target = predict.view(-1, 101 * 101), target.view(-1, 101 * 101)
    intersection = torch.sum((predict > 0.5) * (target == 1), dim=1)
    union = torch.sum(predict > 0.5, dim=1) + torch.sum(target == 1, dim=1) - intersection
    iou = intersection.float() / union.float()
    iou[torch.isnan(iou)] = 1
    return iou_to_score(iou)


def val_score(models: List, val_dl: DataLoader, reverse_tta: List) -> float:
    """
    :param val_dl: each batch: tta list of [img, mask]
    :return: models' score for val_dl
    """
    scores = []
    with torch.no_grad():
        for model in models:
            model.eval()
        for tta_batch in val_dl:
            p_masks = []
            for model in models:
                predicts = []
                for img, mask in tta_batch:
                    predict = model(img)
                    predicts.append(predict)
                p_mask = tta_mean_predict(predicts, reverse_tta)
                p_masks.append(p_mask)
            p_mask = torch.stack(p_masks).mean(dim=0)
            scores.append(get_score(p_mask, mask))
    res = torch.cat(scores).mean().item()
    return res
