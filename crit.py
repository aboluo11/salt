from lightai.imps import *

class Crit:
    def __init__(self, weight):
        """weight: a list, all items sum to 1"""
        self.weight = weight

    def __call__(self, predict, target):
        loss_f = nn.BCEWithLogitsLoss()
        logit_mask, logit_has_salt = predict
        t_mask, t_has_salt = target
        mask_loss = loss_f(logit_mask, t_mask)
        has_salt_loss = loss_f(logit_has_salt, t_has_salt)
        return torch.sum(loss*w for loss,w in zip([mask_loss,has_salt_loss], self.weight))