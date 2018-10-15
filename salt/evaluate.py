from lightai.core import *
from .predict import *


class Evaluator:
    def __init__(self, val_dl, metrics: List, model, loss_fn, reverse_ttas):
        self.val_dl = val_dl
        self.metrics = metrics
        self.model = model
        self.loss_fn = loss_fn
        self.reverse_ttas = reverse_ttas

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
                predict = tta_mean_predict(predicts, self.reverse_ttas)
                for metric in self.metrics:
                    metric(predict, target)
        loss = np.average(losses, weights=bses)
        eval_res = [loss] + [metric.res() for metric in self.metrics]
        return eval_res