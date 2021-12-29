#!/usr/bin/env python3
import unittest
from collections import OrderedDict
from typing import cast

import torch

import captum.optim._core.output_hook as output_hook
from captum.optim.models import googlenet
from tests.helpers.basic import BaseTest


def _count_forward_hooks(model: torch.nn.Module, hook_name: Optional[str] = None) -> int:
    """
    Count the number of active forward hooks on the specified model or module.
    By default nn.Module instance do not have a "_forward_hooks" attribute, and
    we can only remove the hooks without their handles by setting them to their
    default of 'OrderedDict()'. So just because a module has the right attribute,
    doesn't mean that we need to count it.

    Args:

        model (nn.Module): The model instance or target module instance to count
            the number of forward hooks on.
        name (str, optional): Optionally only remove specific forward hooks based on
            their function's __name__ attribute.
            Default: None

    Returns:
        num_hooks (int): The number of active hooks in the specified module.
    """

    num_hooks = 0
    for name, child in model._modules.items():
        if child is not None:
            if hasattr(child, "_forward_hooks"):
                if child._forward_hooks != OrderedDict():
                    if hook_name is not None:
                        dict_items = list(child._forward_hooks.items())
                        for i, fn in dict_items:
                            if fn.__name__ == hook_name:
                                num_hooks +=1
                    else:
                        num_hooks +=1
            _count_forward_hooks(child, hook_name)
    if hasattr(model, "_forward_hooks"):
        if model._forward_hooks != OrderedDict():
            if model._forward_hooks != OrderedDict():
                if hook_name is not None:
                    dict_items = list(model._forward_hooks.items())
                    for i, fn in dict_items:
                        if fn.__name__ == hook_name:
                            num_hooks +=1
                else:
                    num_hooks +=1
    return num_hooks


class TestModuleOutputsHook(BaseTest):
    def test_init_hook_duplication_fix(self) -> None:
        model = torch.nn.Sequential(*[torch.nn.Identity()] * 2)
        for i in range(5):
            _ = output_hook.ModuleOutputsHook([model[1]])
        n_hooks = _count_forward_hooks(model)
        self.assertEqual(n_hooks, 1)


class TestActivationFetcher(BaseTest):
    def test_activation_fetcher(self) -> None:
        if torch.__version__ <= "1.2.0":
            raise unittest.SkipTest(
                "Skipping ActivationFetcher test due to insufficient Torch version."
            )
        model = googlenet(pretrained=True)

        catch_activ = output_hook.ActivationFetcher(model, targets=[model.mixed4d])
        activ_out = catch_activ(torch.zeros(1, 3, 224, 224))

        self.assertIsInstance(activ_out, dict)
        m4d_activ = activ_out[model.mixed4d]
        self.assertEqual(list(cast(torch.Tensor, m4d_activ).shape), [1, 528, 14, 14])


class TestRemoveAllForwardHooks(BaseTest):
    def test_forward_hook_removal(self) -> None:
        def forward_hook(self, input: Tuple[torch.Tensor], output: torch.Tensor) -> None:
            pass
        layer = torch.nn.Sequential(*[torch.nn.Identity()] * 2)
        model = torch.nn.Sequential(*[layer]*2)

        model.register_forward_hook(forward_hook)
        model[1].register_forward_hook(forward_hook)
        model[0][1].register_forward_hook(forward_hook)

        n_hooks = _count_forward_hooks(model)
        self.assertEqual(n_hooks, 3)

        output_hook._remove_all_forward_hooks(model)
        n_hooks = _count_forward_hooks(model)
        self.assertEqual(n_hooks, 0)
