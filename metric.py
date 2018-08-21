from lightai.imps import *
from functional import *

class Score:
    def __init__(self):
        self.scores = []

    def __call__(self, predict, target):
        s = score(torch.sigmoid(predict),target,0.5)
        self.scores.append(s)

    def res(self):
        return torch.cat(self.scores).mean().item()