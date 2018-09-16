from lightai.imps import *
from lightai.callback import CallBack

class Gradient_cb(CallBack):
    def __init__(self, model):
        self.layers = {}
        self.gradients = {}
        for part in [model.encoder, model.upmodel]:
            for layer_name,layer in part._modules.items():
                self.layers[layer_name] = layer
        self.layers['final_conv'] = model.final_conv
        for layer_name in self.layers.keys():
            self.gradients[layer_name] = []

    def on_batch_end(self, loss, model):
        for layer_name, layer in self.layers.items():
            self.gradients[layer_name].append(grad_mean(layer))

    def plot(self):
        _, axes = plt.subplots(len(self.layers), 1, figsize=(15, 5*len(self.layers)))
        for (layer_name,gradient),ax in zip(self.gradients.items(),axes):
            ax.plot(gradient)
            ax.set_title(layer_name)