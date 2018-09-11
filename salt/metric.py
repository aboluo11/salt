from .predict import *


class Score:
    """
    support tta
    """

    def __init__(self):
        self.scores = []

    def __call__(self, predicts, target: torch.Tensor, reverse_tta: Optional[List] = None):
        """
        :param predicts: list of predicts or predict if no tta.
            predict: tta [mask, has_salt], all items are logit tensor
        :param target: mask, logit tensor
        :param reverse_tta: list of function apply to predicted mask in predict or None.
            If None: no tta.
        calculate a batch of predicts' scores, shape: [batch_size], and add it to self.scores
        """
        if not reverse_tta:
            predicts = [predicts]
            reverse_tta = [None]
        p_mask = tta_mean_predict(predicts, reverse_tta)
        s = get_score(p_mask, target, 0.5)
        self.scores.append(s)

    def res(self) -> float:
        """
        :return:  batches of predicts' score, reduced
        """
        return torch.cat(self.scores).mean().item()


def iou_to_score(iou):
    shape = iou.shape
    iou = iou.view(*shape, 1).expand(*shape, 10)
    metric = torch.arange(0.5, 1, 0.05, device='cuda')
    return torch.sum(iou > metric, dim=-1).float() / 10

def get_score(predict: torch.Tensor, target: torch.Tensor, threshold: float) -> torch.Tensor:
    """
    :param predict: batch of predict masks
    :param target: batch of target masks
    :return: a batch of images' score at threshold t, shape: [batch_size]
    """
    predict, target = predict.view(-1, 101 * 101), target.view(-1, 101 * 101)
    metric = torch.arange(0.5, 1, 0.05, device='cuda')
    intersection = torch.sum((predict > threshold) * (target == 1), dim=1)
    union = torch.sum(predict > threshold, dim=1) + torch.sum(target == 1, dim=1) - intersection
    iou = intersection.float() / union.float()
    iou[torch.isnan(iou)] = 1
    iou = iou.view(-1, 1)
    iou = iou.expand(-1, 10)
    return torch.sum(iou > metric, dim=1).float() / 10


def val_score(model: nn.Module, val_dl: DataLoader, reverse_tta: List) -> float:
    """
    :param val_dl: each batch: tta list of [img, mask]
    :return: model's score for val_dl
    """
    metric = Score()
    with torch.no_grad():
        model.eval()
        for tta_batch in val_dl:
            predicts = []
            for img, mask in tta_batch:
                img, mask = T(img), T(mask)
                predict = model(img)
                predicts.append(predict)
            metric(predicts, mask, reverse_tta)
    return metric.res()
