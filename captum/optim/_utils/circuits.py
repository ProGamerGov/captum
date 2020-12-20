import torch
import torch.nn as nn

from captum.optim._utils.models import collect_activations


def get_expanded_weights(
    model,
    target1: nn.Module,
    target2: nn.Module,
    model_input: torch.Tensor = torch.zeros(1, 3, 224, 224),
) -> torch.Tensor:
    """
    Extract meaningful weight interactions from between neurons which aren’t
    literally adjacent in a neural network, or where the weights aren’t directly
    represented in a single weight tensor.

    Schubert, et al., "Visualizing Weights", Distill, 2020.
    See: https://distill.pub/2020/circuits/visualizing-weights/
    """

    activations = collect_activations(model, [target1, target2], model_input)
    activ1 = activations[target1]
    activ2 = activations[target2]

    if activ2.dim() == 4:
        t_offset_h, t_offset_w = (activ2.size(2) - 1) // 2, (activ2.size(3) - 1) // 2
        t_center = activ2[:, :, t_offset_h, t_offset_w]
    elif activ2.dim() == 2:
        t_center = activ2

    A = []
    for i in range(activ2.size(1)):
        x = torch.autograd.grad(
            outputs=t_center[:, i],
            inputs=[activ1],
            grad_outputs=torch.ones_like(t_center[:, i]),
            retain_graph=True,
        )[0]
        A.append(x)
    return torch.stack(A, -1)[0]


def remove_constant_pad(x: torch.Tensor) -> torch.Tensor:
    """
    Remove constant padding from an NCHW tensor's H & W dimensions.


    Designed to automatically crop outputs from get_expanded_weights()
    """

    assert x.dim() == 3 or x.dim() == 4
    x = x.unsqueeze(0) if x.dim() == 3 else x

    pad = torch.where(x[0, 0] - x[0, 0, 0, 0] != 0)[0][0]

    x_pad, y_pad = 0, 0
    if (x.size(3) % 2) == 0:
        x_pad = 1
    if (x.size(2) % 2) == 0:
        y_pad = 1
    return x[:, :, pad : -(pad + y_pad), pad : -(pad + x_pad)]
