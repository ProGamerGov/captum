import warnings
from collections import OrderedDict
from typing import Callable, Dict, Iterable, Tuple
from warnings import warn

import torch
import torch.nn as nn

from captum.optim._utils.typing import ModuleOutputMapping, TupleOfTensorsOrTensorType


class ModuleOutputsHook:
    def __init__(self, target_modules: Iterable[nn.Module]) -> None:
        """
        Args:

            target_modules (Iterable of nn.Module): A list of nn.Module targets.
        """
        for module in target_modules:
            _remove_all_forward_hooks(module, "module_outputs_forward_hook")
        self.outputs: ModuleOutputMapping = dict.fromkeys(target_modules, None)
        self.hooks = [
            module.register_forward_hook(self._forward_hook())
            for module in target_modules
        ]

    def _reset_outputs(self) -> None:
        """
        Delete captured activations.
        """
        self.outputs = dict.fromkeys(self.outputs.keys(), None)

    @property
    def is_ready(self) -> bool:
        return all(value is not None for value in self.outputs.values())

    def _forward_hook(self) -> Callable:
        """
        Return the forward_hook function.

        Returns:
            forward_hook (Callable): The forward_hook function.
        """

        def module_outputs_forward_hook(
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

        return module_outputs_forward_hook

    def consume_outputs(self) -> ModuleOutputMapping:
        """
        Collect target activations and return them.

        Returns:
            outputs (ModuleOutputMapping): The captured outputs.
        """
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
        """
        Remove hooks.
        """
        for hook in self.hooks:
            hook.remove()

    def __del__(self) -> None:
        """
        Ensure that using 'del' properly deletes hooks.
        """
        self.remove_hooks()


class ActivationFetcher:
    """
    Simple module for collecting activations from model targets.
    """

    def __init__(self, model: nn.Module, targets: Iterable[nn.Module]) -> None:
        """
        Args:

            model (nn.Module):  The reference to PyTorch model instance.
            targets (nn.Module or list of nn.Module):  The target layers to
                collect activations from.
        """
        super(ActivationFetcher, self).__init__()
        self.model = model
        self.layers = ModuleOutputsHook(targets)

    def __call__(self, input_t: TupleOfTensorsOrTensorType) -> ModuleOutputMapping:
        """
        Args:

            input_t (tensor or tuple of tensors, optional):  The input to use
                with the specified model.

        Returns:
            activations_dict: An dict containing the collected activations. The keys
                for the returned dictionary are the target layers.
        """

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.model(input_t)
            activations_dict = self.layers.consume_outputs()
        finally:
            self.layers.remove_hooks()
        return activations_dict


def _remove_all_forward_hooks(
    module: torch.nn.Module, hook_name: Optional[str] = None
) -> None:
    """
    This function removes all forward hooks in the specified module, without requiring
    any hook handles. This lets us clean up & remove any hooks that weren't property
    deleted.
    
    Warning: Various PyTorch modules and systems make use of hooks, and thus extreme
    caution should be exercised when removing all hooks. Users are recommended to give
    their hook function a unique name that can be used to safetly identify and remove
    the hook.

    Args:

        module (nn.Module): The module instance to remove forward hooks from.
        name (str, optional): Optionally only remove specific forward hooks based on
            their function's __name__ attribute.
            Default: None
    """
    if hook_name is None or hook_name == "":
        warn(
            "Warning modules like weight_norm will be broken by removing all hooks."
            + " Please specify the full & unique function name of the target hook to"
            + " avoid errors"
        )

    # Remove hooks from target submodules
    for name, child in module._modules.items():
        if child is not None:
            if hasattr(child, "_forward_hooks"):
                if child._forward_hooks != OrderedDict():
                    if hook_name is not None:
                        dict_items = list(child._forward_hooks.items())
                        child._forward_hooks = OrderedDict(
                            [
                                (i, fn)
                                for i, fn in dict_items
                                if fn.__name__ != hook_name
                            ]
                        )
                    else:
                        child._forward_hooks: Dict[int, Callable] = OrderedDict()
            _remove_all_forward_hooks(child, hook_name)

    # Remove hooks from the target module
    if hasattr(module, "_forward_hooks"):
        if module._forward_hooks != OrderedDict():
            if module._forward_hooks != OrderedDict():
                if hook_name is not None:
                    dict_items = list(module._forward_hooks.items())
                    module._forward_hooks = OrderedDict(
                        [(i, fn) for i, fn in dict_items if fn.__name__ != hook_name]
                    )
                else:
                    module._forward_hooks: Dict[int, Callable] = OrderedDict()
