from lightai.imps import *
from functional import *

class Score:
    def __init__(self):
        self.scores = []

    def __call__(self, predict, target):
        batch = []
        predict = torch.sigmoid(predict)
        for t in np.linspace(0.4,0.6,num=21,endpoint=True):
            s = score(predict,target,t)
            batch.append(s)
        self.scores.append(torch.stack(batch))

    def res(self):
        return torch.cat(self.scores, dim=1).mean(dim=1).max().item()