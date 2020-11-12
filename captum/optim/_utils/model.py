import torch
import torch.nn as nn
import torch.nn.functional as F


def get_model_layers(model):
    """
    Return a list of hookable layers for the target model.
    """
    layers = []

    def get_layers(net, prefix=[]):
        if hasattr(net, "_modules"):
            for name, layer in net._modules.items():
                if layer is None:
                    continue
                layers.append(".".join(prefix + [name]))
                get_layers(layer, prefix=prefix + [name])

    get_layers(model)
    return layers


# RedirectedReLU autograd function
class RedirectedReLU(torch.autograd.Function):
    @staticmethod
    def forward(self, input_tensor):
        self.save_for_backward(input_tensor)
        return input_tensor.clamp(min=0)

    @staticmethod
    def backward(self, grad_output):
        (input_tensor,) = self.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input_tensor < 0] = grad_input[input_tensor < 0] * 1e-1
        return grad_input


# RedirectedReLU layer
class RedirectedReluLayer(nn.Module):
    def forward(self, input):
        if F.relu(input.detach().sum()) != 0:
            return F.relu(input, inplace=True)
        else:
            return RedirectedReLU.apply(input)


# Basic Hookable & Replaceable ReLU layer
class ReluLayer(nn.Module):
    def forward(self, input):
        return F.relu(input, inplace=True)


# Replace all target layers
def replace_layer(model, layer1=ReluLayer, layer2=RedirectedReluLayer()):
    for name, child in model.named_children():
        if isinstance(child, layer1):
            setattr(model, name, layer2)
        else:
            relu_to_redirected_relu(child)


# Basic Hookable Local Response Norm layer
class LocalResponseNormLayer(nn.Module):
    def __init__(self, size=5, alpha=9.999999747378752e-05, beta=0.75, k=1):
        super(LocalResponseNormLayer, self).__init__()
        self.size = size
        self.alpha = alpha
        self.beta = beta
        self.k = k

    def forward(self, input):
        return F.local_response_norm(
            input, size=self.size, alpha=self.alpha, beta=self.beta, k=self.k
        )
