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
        if self.iter % 10 == 0:
            for layer_name, layer in self.layers.items():
                weights = []
                children = leaves(layer)
                for each in children:
                    if hasattr(each, 'weight') and each.__class__.__name__[:9] != 'BatchNorm':
                        weights.append(each.weight.view(-1))
                weight = torch.cat(weights).cpu().detach().numpy()
                self.writer.add_histogram(f'{layer_name}_weight', weight, self.iter)
        self.iter += 1

