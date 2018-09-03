from lightai.imps import *

class Crit:
    def __init__(self, weight):
        """weight: a list, all items sum to 1"""
        self.weight = weight
        self.loss_f = nn.BCEWithLogitsLoss()

    def __call__(self, predict, target):
        """
        predict: [mask, has_salt(logit)]
        target: mask
        """
        p_mask, p_has_salt = predict
        has_salt_index = target.byte().view(target.shape[0],-1).any(dim=1)
        t_has_salt = has_salt_index.float()
        has_salt_loss = self.loss_f(p_has_salt, t_has_salt)
        if has_salt_index.any():
            mask_loss = self.loss_f(p_mask[has_salt_index], target[has_salt_index])
        else:
            mask_loss = 0
        return mask_loss*self.weight[0] + has_salt_loss*self.weight[1]