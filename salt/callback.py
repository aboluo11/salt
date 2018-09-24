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
            weight_means = []
            weight_stds = []
            children = leaves(layer)
            for each in children:
                if hasattr(each, 'weight') and each.__class__.__name__[:9] != 'BatchNorm':
                    weight_means.append(each.weight.mean().item())
                    weight_stds.append(each.weight.std().item())
            weight_mean = np.array(weight_means).mean()
            weight_std = np.array(weight_stds).mean()
            self.writer.add_scalar(f'{layer_name}_weight_mean', weight_mean, self.iter)
            self.writer.add_scalar(f'{layer_name}_weight_std', weight_std, self.iter)
        self.iter += 1

