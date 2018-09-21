from lightai.imps import *
from lightai.callback import CallBack
from tensorboardX import SummaryWriter

class GradientLogger(CallBack):
    def __init__(self, model, writer):
        self.layers = {}
        for part in [model.encoder, model.upmodel]:
            for layer_name,layer in part._modules.items():
                self.layers[layer_name] = layer
        self.layers['final_conv'] = model.final_conv
        self.iter = 0
        self.writer = writer

    def on_batch_end(self, loss, model):
        for layer_name, layer in self.layers.items():
            grad_mean = []
            grad_std = []
            children = leaves(layer)
            for each in children:
                if hasattr(each, 'weight') and each.__class__.__name__[:9] != 'BatchNorm':
                    grad_mean.append(each.weight.grad.mean().item())
                    grad_std.append(each.weight.grad.std().item())
            grad_mean = np.array(grad_mean).mean()
            grad_std = np.array(grad_std).mean()
            self.writer.add_scalar(f'{layer_name}_grad_mean', grad_mean, self.iter)
            self.writer.add_scalar(f'{layer_name}_grad_std', grad_std, self.iter)
        self.iter += 1

