from lightai.imps import *
from functional import *

class Score:
    def __init__(self):
        self.scores = []

    def __call__(self, predict, target):
        """
        predict: [mask, has_salt(logit)]
        target: mask
        """
        batch = []
        p_mask = predict[0]
        has_salt_index = torch.sigmoid(predict[1]) > 0.5
        if has_salt_index.any():
            p_mask[has_salt_index] = torch.sigmoid(p_mask[has_salt_index])
        if (~has_salt_index).any():
            p_mask[~has_salt_index] = torch.zeros_like(p_mask[0], dtype=torch.float32, device='cuda')
        for t in np.linspace(0.4,0.6,num=21,endpoint=True):
            s = score(p_mask,target,t)
            batch.append(s)
        self.scores.append(torch.stack(batch))

    def res(self):
        return torch.cat(self.scores, dim=1).mean(dim=1).max().item()