from lightai.imps import *


class Crit:
    def __init__(self, mask_loss, weight):
        """weight: a list, all items sum to 1"""
        self.weight = weight
        self.has_salt_loss = nn.BCEWithLogitsLoss()
        self.mask_loss = mask_loss

    def __call__(self, predict, target):
        """
        predict: [mask(logit), has_salt(logit)]
        target: mask
        """
        bs = target.shape[0]
        p_mask, p_has_salt = predict
        has_salt_index = target.byte().view(bs, -1).any(dim=1)
        t_has_salt = has_salt_index.float()
        has_salt_loss = self.has_salt_loss(p_has_salt, t_has_salt)
        if has_salt_index.any():
            mask_loss = self.mask_loss(p_mask[has_salt_index], target[has_salt_index])
        else:
            mask_loss = 0
        return mask_loss * has_salt_index.sum().float() / bs * self.weight[0] + has_salt_loss * self.weight[1]


def get_weight(gt_sorted):
    gt_sorted = gt_sorted.byte()
    gts = gt_sorted.sum(dim=1, keepdim=True)
    intersection = gts - gt_sorted.cumsum(1)
    union = gts + (1 - gt_sorted).cumsum(1)
    iou_loss = 1 - intersection.float() / union.float()
    p = gt_sorted.shape[-1]
    weight[:, 1:p] = iou_loss[:, 1:p] - iou_loss[:, 0:-1]
    return weight


def lovasz(logit, target):
    bs = logit.shape[0]
    logit = logit.view(bs, -1)
    target = target.view(bs, -1)
    length = target.shape[-1]
    sign = 2 * target - 1
    error = 1 - logit * sign
    error_sorted, perm = torch.sort(error, dim=-1, descending=True)
    gt_sorted = target.gather(dim=1, index=perm)
    weight = get_weight(gt_sorted)
    loss = torch.bmm(torch.relu(error_sorted).view(bs, 1, -1), weight.view(bs, -1, 1))
    return loss.mean()
