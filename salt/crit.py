from lightai.imps import *
from salt.metric import iou_to_score

class Crit:
    def __init__(self, mask_loss, weight):
        self.weight = weight
        # self.has_salt_loss = nn.BCEWithLogitsLoss()
        self.mask_loss = mask_loss

    def __call__(self, predict, target):
        """
        predict: [mask(logit) if at least one target has salt, else None, has_salt(logit)]
        target: mask
        """
        # bs = target.shape[0]
        logit_pixel = predict
        # t_has_salt_index = target.byte().view(bs, -1).any(dim=1)
        # logit_img_loss = self.has_salt_loss(logit_img, t_has_salt_index.float())
        # if t_has_salt_index.any():
        #     logit_pixel_loss = self.mask_loss(logit_pixel[t_has_salt_index], target[t_has_salt_index])
        # else:
        #     logit_pixel_loss = 0
        logit_loss = self.mask_loss(logit_pixel, target)
        return logit_loss


def get_weight(gt_sorted):
    gt_sorted = gt_sorted.byte()
    gts = gt_sorted.sum(dim=1, keepdim=True)
    intersection = gts - gt_sorted.cumsum(1)
    union = gts + (1 - gt_sorted).cumsum(1)
    iou = intersection.float() / union.float()
    delta = 1 - iou
    delta[:, 1:] = delta[:, 1:] - delta[:, :-1]
    return delta


def lovasz(logit, target):
    bs = logit.shape[0]
    logit = logit.view(bs, -1)
    target = target.view(bs, -1)
    sign = 2 * target - 1
    error = torch.relu(1 - logit * sign)
    error_sorted, perm = torch.sort(error, dim=-1, descending=True)
    gt_sorted = target.gather(dim=1, index=perm)
    weight = get_weight(gt_sorted)
    delta_average = torch.bmm(error_sorted.view(bs, 1, -1), weight.view(bs, -1, 1))
    return delta_average.mean()
