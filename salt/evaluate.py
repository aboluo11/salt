from lightai.core import *


class Evaluator:
    def __init__(self, val_dl, metric, model, loss_fn):
        self.val_dl = val_dl
        self.metric = metric
        self.model = model
        self.loss_fn = loss_fn

    def __call__(self):
        losses, bses = [], []
        self.model.eval()
        with torch.no_grad():
            for tta_batch in self.val_dl:
                predicts = []
                for i, (x, target) in enumerate(tta_batch):
                    predict = self.model(x)
                    predicts.append(predict)
                    if i == 0:
                        loss = self.loss_fn(predict, target)
                        losses.append(loss.item())
                        bses.append(len(target))
                self.metric(predicts, target)
        loss = np.average(losses, weights=bses)
        return loss, self.metric.res()
