#!/usr/bin/env python3
import unittest
from collections import OrderedDict
from typing import List, Optional, Tuple, cast

import torch

import captum.optim._core.output_hook as output_hook
from captum.optim.models import googlenet
from tests.helpers.basic import BaseTest


def _count_forward_hooks(
    module: torch.nn.Module, hook_fn_name: Optional[str] = None
) -> int:
    """
    Count the number of active forward hooks on the specified module instance.

    Args:

        module (nn.Module): The model module instance to count the number of
            forward hooks on.
        name (str, optional): Optionally only count specific forward hooks based on
            their function's __name__ attribute.
            Default: None

    Returns:
        num_hooks (int): The number of active hooks in the specified module.
    """

    num_hooks: List[int] = [0]

    def _count_hooks(m: torch.nn.Module, name: Optional[str] = None) -> None:
        if hasattr(m, "_forward_hooks"):
            if m._forward_hooks != OrderedDict():
                dict_items = list(m._forward_hooks.items())
                for i, fn in dict_items:
                    if hook_fn_name is None or fn.__name__ == name:
                        num_hooks[0] += 1

    def _count_child_hooks(
        target_module: torch.nn.Module,
        hook_name: Optional[str] = None,
    ) -> None:

        for name, child in target_module._modules.items():
            if child is not None:
                _count_hooks(child, hook_name)
                _count_child_hooks(child, hook_name)

    _count_child_hooks(module, hook_fn_name)
    _count_hooks(module, hook_fn_name)
    return num_hooks[0]


class TestModuleOutputsHook(BaseTest):
    def test_init_hook_duplication_fix(self) -> None:
        model = torch.nn.Sequential(torch.nn.Identity(), torch.nn.Identity())
        for i in range(5):
            _ = output_hook.ModuleOutputsHook([model[1]])
        n_hooks = _count_forward_hooks(model, "module_outputs_forward_hook")
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
        def forward_hook_unique_fn(
            self, input: Tuple[torch.Tensor], output: torch.Tensor
        ) -> None:
            pass

        layer1 = torch.nn.Sequential(torch.nn.Identity(), torch.nn.Identity())
        layer2 = torch.nn.Sequential(torch.nn.Identity(), torch.nn.Identity())
        model = torch.nn.Sequential(layer1, layer2)

        model.register_forward_hook(forward_hook_unique_fn)
        model[1].register_forward_hook(forward_hook_unique_fn)
        model[0][1].register_forward_hook(forward_hook_unique_fn)

        n_hooks = _count_forward_hooks(model, "forward_hook_unique_fn")
        self.assertEqual(n_hooks, 3)

        output_hook._remove_all_forward_hooks(model, "forward_hook_unique_fn")
        n_hooks = _count_forward_hooks(model)
        self.assertEqual(n_hooks, 0)

    def test_forward_hook_removal_unique_fn(self) -> None:
        def forward_hook_unique_fn_1(
            self, input: Tuple[torch.Tensor], output: torch.Tensor
        ) -> None:
            pass

        def forward_hook_unique_fn_2(
            self, input: Tuple[torch.Tensor], output: torch.Tensor
        ) -> None:
            pass

        layer1 = torch.nn.Sequential(torch.nn.Identity(), torch.nn.Identity())
        layer2 = torch.nn.Sequential(torch.nn.Identity(), torch.nn.Identity())
        model = torch.nn.Sequential(layer1, layer2)

        model.register_forward_hook(forward_hook_unique_fn_1)
        model[1].register_forward_hook(forward_hook_unique_fn_1)
        model[0][1].register_forward_hook(forward_hook_unique_fn_1)

        model.register_forward_hook(forward_hook_unique_fn_2)
        model[1][0].register_forward_hook(forward_hook_unique_fn_2)

        n_hooks = _count_forward_hooks(model, "forward_hook_unique_fn_1")
        self.assertEqual(n_hooks, 3)
        n_hooks = _count_forward_hooks(model, "forward_hook_unique_fn_2")
        self.assertEqual(n_hooks, 2)

        n_hooks = _count_forward_hooks(model)
        self.assertEqual(n_hooks, 5)

        output_hook._remove_all_forward_hooks(model, "forward_hook_unique_fn_1")
        n_hooks = _count_forward_hooks(model)
        self.assertEqual(n_hooks, 2)

        output_hook._remove_all_forward_hooks(model, "forward_hook_unique_fn_2")
        n_hooks = _count_forward_hooks(model)
        self.assertEqual(n_hooks, 0)
