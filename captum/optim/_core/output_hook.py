import warnings
from typing import Callable, Iterable, Tuple
from warnings import warn

import torch
import torch.nn as nn

from captum.optim._utils.typing import ModuleOutputMapping, TupleOfTensorsOrTensorType


class ModuleReuseException(Exception):
    pass


class ModuleOutputsHook:
    def __init__(self, target_modules: Iterable[nn.Module]) -> None:
        self.outputs: ModuleOutputMapping = dict.fromkeys(target_modules, None)
        self.hooks = [
            module.register_forward_hook(self._forward_hook())
            for module in target_modules
        ]

    def _reset_outputs(self) -> None:
        self.outputs = dict.fromkeys(self.outputs.keys(), None)

    @property
    def is_ready(self) -> bool:
        return all(value is not None for value in self.outputs.values())

    def _forward_hook(self) -> Callable:
        def forward_hook(
            module: nn.Module, input: Tuple[torch.Tensor], output: torch.Tensor
        ) -> None:
            assert module in self.outputs.keys()
            if self.outputs[module] is None:
                self.outputs[module] = output
            else:
                warn(
                    f"Hook attached to {module} was called multiple times. "
                    "As of 2019-11-22 please don't reuse nn.Modules in your models."
                )
            if self.is_ready:
                warn(
                    "No outputs found from models. This can be ignored if you are "
                    "optimizing on inputs only, without models. Otherwise, check "
                    "that you are passing model layers in your losses."
                )

        return forward_hook

    def consume_outputs(self) -> ModuleOutputMapping:
        if not self.is_ready:
            warn(
                "Consume captured outputs, but not all requested target outputs "
                "have been captured yet!"
            )
        outputs = self.outputs
        self._reset_outputs()
        return outputs

    @property
    def targets(self) -> Iterable[nn.Module]:
        return self.outputs.keys()

    def remove_hooks(self) -> None:
        for hook in self.hooks:
            hook.remove()

    def __del__(self) -> None:
        # print(f"DEL HOOKS!: {list(self.outputs.keys())}")
        self.remove_hooks()


class ActivationFetcher:
    """
    Simple module for collecting activations from model targets.
    """

    def __init__(self, model: nn.Module, targets: Iterable[nn.Module]) -> None:
        super(ActivationFetcher, self).__init__()
        self.model = model
        self.layers = ModuleOutputsHook(targets)

    def __call__(self, input_t: TupleOfTensorsOrTensorType) -> ModuleOutputMapping:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.model(input_t)
            activations = self.layers.consume_outputs()
        finally:
            self.layers.remove_hooks()
        return activations
