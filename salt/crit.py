from lightai.core import *
from salt.metric import iou_to_score

class Crit:
    def __init__(self, weight):
        self.weight = weight
        self.lovasz = lovasz
        self.bce = nn.BCEWithLogitsLoss()

    def __call__(self, predict, target):
        """
        predict: [mask(logit) if at least one target has salt, else None, has_salt(logit)]
        target: mask
        """
        bs = target.shape[0]
        logit, logit_pixel, logit_img = predict
        t_has_salt_index = target.byte().view(bs, -1).any(dim=1)
        logit_img_loss = self.bce(logit_img, t_has_salt_index.float())
        logit_pixel_loss = 0
        has_salt_sum = t_has_salt_index.sum()
        if t_has_salt_index.any():
            weight = has_salt_sum.float() / bs
            logit_pixel_loss = self.lovasz(logit_pixel[t_has_salt_index], target[t_has_salt_index]) * weight
        logit_loss = self.lovasz(logit, target)
        return logit_loss * self.weight[0] + logit_pixel_loss * self.weight[1] + logit_img_loss * self.weight[2]


def get_weight(gt_sorted, error_sorted):
    gt_sorted = gt_sorted.byte()
    gts = gt_sorted.sum(dim=1, keepdim=True)

    weight = torch.zeros_like(gt_sorted, dtype=torch.float32)

    empty_index = (gts.view(-1) == 0)
    if empty_index.any():
        error_sum = error_sorted.sum(dim=1, keepdim=True)
        idx = empty_index * (error_sum.view(-1) != 0)
        weight[idx] = error_sorted[idx]/error_sum[idx]

    non_empty_index = ~empty_index
    if non_empty_index.any():
        intersection = gts[non_empty_index] - gt_sorted[non_empty_index].cumsum(1)
        union = gts[non_empty_index] + (1 - gt_sorted[non_empty_index]).cumsum(1)
        iou = intersection.float() / union.float()
        delta = 1 - iou
        weight[non_empty_index, 1:] = delta[:, 1:] - delta[:, :-1]
    return weight


def lovasz(logit, target):
    bs = logit.shape[0]
    logit = logit.view(bs, -1)
    target = target.view(bs, -1)
    sign = 2 * target - 1
    error = torch.relu(1 - logit * sign)
    error_sorted, perm = torch.sort(error, dim=-1, descending=True)
    gt_sorted = target.gather(dim=1, index=perm)
    weight = get_weight(gt_sorted, error_sorted)
    delta_average = torch.bmm(error_sorted.view(bs, 1, -1), weight.view(bs, -1, 1))
    return delta_average.mean()
