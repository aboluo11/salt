from lightai.imps import *
from lightai.callback import CallBack

class Gradient_cb(CallBack):
    def __init__(self, model):
        self.layers = {}
        self.gradients = {}
        self.weights = {}
        for part in [model.encoder, model.upmodel]:
            for layer_name,layer in part._modules.items():
                self.layers[layer_name] = layer
        self.layers['final_conv'] = model.final_conv
        for layer_name in self.layers.keys():
            self.gradients[layer_name] = []
            self.weights[layer_name] = []

    def on_batch_end(self, loss, model):
        for layer_name, layer in self.layers.items():
            weight, grad = weight_grad_mean(layer)
            self.gradients[layer_name].append(grad)
            self.weights[layer_name].append(weight)

    def plot(self):
        _, axes = plt.subplots(len(self.layers), 2, figsize=(15, 5*len(self.layers)))
        for (layer_name,gradient),weight,ax in zip(self.gradients.items(),self.weights.values(),axes):
            ax[0].plot(gradient)
            ax[1].plot(weight)
            ax[0].set_title(f'{layer_name}, gradient')
            ax[1].set_title(f'{layer_name}, weight')
