from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from captum.optim._utils.typing import ModuleOutputMapping


class Loss(ABC):
    """
    Abstract Class to describe loss.
    """

    def __init__(self, target: nn.Module) -> None:
        super(Loss, self).__init__()
        self.target = target

    @abstractmethod
    def __call__(self, x):
        pass

    def get_neuron_pos(self, H: int, W: int, x: int =None, y: int=None):
        if x is None:
            _x = W // 2
        else:
            assert x < W
            _x = x

        if y is None:
            _y = H // 2
        else:
            assert y < W
            _y = y
        return _x, _y


class LayerActivation(Loss):
    """
    Maximize activations at the target layer.
    """

    def __call__(self, targets_to_values: ModuleOutputMapping) -> torch.Tensor:
        return targets_to_values[self.target]


class ChannelActivation(Loss):
    """
    Maximize activations at the target layer and target channel.
    """

    def __init__(self, target: nn.Module, channel_index: int) -> None:
        super(Loss, self).__init__()
        self.target = target
        self.channel_index = channel_index

    def __call__(self, targets_to_values: ModuleOutputMapping) -> torch.Tensor:
        activations = targets_to_values[self.target]
        assert activations is not None
        # ensure channel_index is valid
        assert self.channel_index < activations.shape[1]
        # assume NCHW
        # NOTE: not necessarily true e.g. for Linear layers
        # assert len(activations.shape) == 4
        return activations[:, self.channel_index, ...]


class NeuronActivation(Loss):
    def __init__(
        self, target: nn.Module, channel_index: int, x: int = None, y: int = None
    ) -> None:
        super(Loss, self).__init__()
        self.target = target
        self.channel_index = channel_index
        self.x = x
        self.y = y

        # ensure channel_index will be valid
        assert self.channel_index < self.target.out_channels

    def _call__(self, targets_to_values: ModuleOutputMapping) -> torch.Tensor:
        activations = targets_to_values[self.target]
        assert activations is not None
        assert len(activations.shape) == 4  # assume NCHW
        _x, _y = self.get_neuron_pos(
            activations.size(2), activations.size(3), self.x, self.y
        )

        return activations[:, self.channel_index, _x, _y]


class DeepDream(Loss):
    """
    Maximize 'interestingness' at the target layer.
    Mordvintsev et al., 2015.
    """

    def __call__(self, targets_to_values: ModuleOutputMapping) -> torch.Tensor:
        activations = targets_to_values[self.target]
        return activations ** 2


class TotalVariation(Loss):
    """
    Total variation denoising penalty for activations.
    See Simonyan, et al., 2014.
    """

    def _call__(self, targets_to_values: ModuleOutputMapping) -> torch.Tensor:
        activations = targets_to_values[self.target]
        x_diff = activations[..., 1:, :] - activations[..., :-1, :]
        y_diff = activations[..., :, 1:] - activations[..., :, :-1]
        return torch.sum(torch.abs(x_diff)) + torch.sum(torch.abs(y_diff))


class L1(Loss):
    """
    L1 norm of the target layer, generally used as a penalty.
    """

    def __init__(self, target: nn.Module, constant: float = 0) -> None:
        super(Loss, self).__init__()
        self.target = target
        self.constant = constant

    def _call__(self, targets_to_values: ModuleOutputMapping) -> torch.Tensor:
        activations = targets_to_values[self.target]
        return torch.abs(activations - self.constant).sum()


class L2(Loss):
    """
    L2 norm of the target layer, generally used as a penalty.
    """

    def __init__(
        self, target: nn.Module, constant: float = 0, epsilon: float = 1e-6
    ) -> None:
        self.target = target
        self.constant = constant
        self.epsilon = epsilon

    def _call__(self, targets_to_values: ModuleOutputMapping) -> torch.Tensor:
        activations = targets_to_values[self.target]
        activations = (activations - self.constant).sum()
        return torch.sqrt(self.epsilon + activations)


class Diversity(Loss):
    """
    Use a cosine similarity penalty to extract features from a polysemantic neuron.
    Olah, Mordvintsev & Schubert, 2017.
    https://distill.pub/2017/feature-visualization/#diversity
    """

    def _call__(self, targets_to_values: ModuleOutputMapping) -> torch.Tensor:
        activations = targets_to_values[self.target]
        return -sum(
            [
                sum(
                    [
                        (
                            torch.cosine_similarity(
                                activations[j].view(1, -1), activations[i].view(1, -1)
                            )
                        ).sum()
                        for i in range(activations.size(0))
                        if i != j
                    ]
                )
                for j in range(activations.size(0))
            ]
        ) / activations.size(0)


class ActivationInterpolation(Loss):
    """
    Interpolate between two different layers & channels
    """

    def __init__(
        self,
        target1: nn.Module,
        channel_index1: int,
        target2: nn.Module,
        channel_index2: int,
    ) -> None:
        super(Loss, self).__init__()
        self.target_one = target1
        self.channel_index_one = channel_index1 or -1
        self.target_two = target2
        self.channel_index_two = channel_index2 or -1

    def __call__(self, targets_to_values: ModuleOutputMapping) -> torch.Tensor:
        activations_one = targets_to_values[self.target_one]
        activations_two = targets_to_values[self.target_two]

        assert activations_one is not None and activations_two is not None
        # ensure channel indices are valid
        assert (
            self.channel_index_one < activations_one.shape[1]
            and self.channel_index_two < activations_two.shape[1]
        )
        assert activations_one.size(0) == activations_two.size(0)

        if self.channel_index_one > -1:
            activations_one = activations_one[:, self.channel_index_one]
        if self.channel_index_two > -1:
            activations_two = activations_two[:, self.channel_index_two]
        B = activations_one.size(0)

        batch_weights = torch.arange(B, device=activations_one.device) / (B - 1)
        sum_tensor = torch.zeros(1, device=activations_one.device)
        for n in range(B):
            sum_tensor = (
                sum_tensor + ((1 - batch_weights[n]) * activations_one[n]).mean()
            )
            sum_tensor = sum_tensor + (batch_weights[n] * activations_two[n]).mean()
        return sum_tensor


class Alignment(Loss):
    """
    Penalize the L2 distance between tensors in the batch to encourage visual
    similarity between them.
    """

    def __init__(self, target: nn.Module, decay_ratio: float = 2) -> None:
        super(Loss, self).__init__()
        self.target = target
        self.decay_ratio = decay_ratio

    def __call__(self, targets_to_values: ModuleOutputMapping) -> torch.Tensor:
        activations = targets_to_values[self.target]
        B = activations.size(0)

        sum_tensor = torch.zeros(1, device=activations.device)
        for d in [1, 2, 3, 4]:
            for i in range(B - d):
                a, b = i, i + d
                activ_a, activ_b = activations[a], activations[b]
                sum_tensor = sum_tensor + (
                    (activ_a - activ_b) ** 2
                ).mean() / self.decay_ratio ** float(d)

        return sum_tensor


class Direction(Loss):
    """
    Visualize a direction.
    """

    def __init__(self, target: nn.Module, vec: torch.Tensor) -> None:
        super(Loss, self).__init__()
        self.target = target
        self.direction = vec.reshape((1, -1, 1, 1))

    def __call__(self, targets_to_values: ModuleOutputMapping) -> torch.Tensor:
        activations = targets_to_values[self.target]
        return torch.cosine_similarity(self.direction, activations)


class DirectionNeuron(Loss):
    """
    Visualize a neuron direction.
    """

    def __init__(
        self,
        target: nn.Module,
        vec: torch.Tensor,
        channel_index: int,
        x: int = None,
        y: int = None,
    ) -> None:
        super(Loss, self).__init__()
        self.target = target
        self.direction = vec.reshape((1, -1, 1, 1))
        self.channel_index = channel_index
        self.x = x
        self.y = y

    def __call__(self, targets_to_values: ModuleOutputMapping) -> torch.Tensor:
        activations = targets_to_values[self.target]

        assert activations.dim() == 4

        _x, _y = self.get_neuron_pos(
            activations.size(2), activations.size(3), self.x, self.y
        )
        activations = activations[:, self.channel_index, _x, _y]
        return torch.cosine_similarity(self.direction, activations[None, None, None])


class TensorDirection(Loss):
    """
    Visualize a tensor direction.
    """

    def __init__(self, target: nn.Module, vec: torch.Tensor) -> None:
        super(Loss, self).__init__()
        self.target = target
        self.direction = vec

    def __call__(self, targets_to_values: ModuleOutputMapping) -> torch.Tensor:
        activations = targets_to_values[self.target]

        assert activations.dim() == 4

        H_vec, W_vec = self.direction.size(2), self.direction.size(3)
        H_activ, W_activ = activations.size(2), activations.size(3)

        H = (H_activ - W_vec) // 2
        W = (W_activ - W_vec) // 2

        activations = activations[:, :, H : H + H_vec, W : W + W_vec]
        return torch.cosine_similarity(self.direction, activations)


class ActivationWeights(Loss):
    """
    Apply weights to channels, neurons, or spots in the target.
    """

    def __init__(
        self,
        target: nn.Module,
        weights: torch.Tensor = None,
        neuron: bool = False,
        x: int = None,
        y: int = None,
        wx: int = None,
        wy: int = None,
    ) -> None:
        super(Loss, self).__init__()
        self.target = target
        self.x = x
        self.y = y
        self.wx = wx
        self.wy = wy
        self.weights = weights
        self.neuron = x is not None or y is not None or neuron
        assert (
            wx is None
            and wy is None
            or wx is not None
            and wy is not None
            and x is not None
            and y is not None
        )

    def __call__(self, targets_to_values: ModuleOutputMapping) -> torch.Tensor:
        activations = targets_to_values[self.target]
        if self.neuron:
            assert activations.dim() == 4
            if self.wx is None and self.wy is None:
                _x, _y = self.get_neuron_pos(
                    activations.size(2), activations.size(3), self.x, self.y
                )
                activations = activations[..., _x, _y].squeeze() * self.weights
            else:
                activations = activations[
                    ..., self.y : self.y + self.wy, self.x : self.x + self.wx
                ] * self.weights.view(1, -1, 1, 1)
        else:
            activations = activations * self.weights.view(1, -1, 1, 1)
        return activations
