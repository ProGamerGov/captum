#!/usr/bin/env python3
import unittest
from collections import OrderedDict
from typing import List, Optional, Tuple, cast

import torch

import captum.optim._core.output_hook as output_hook
from captum.optim.models import googlenet
from tests.helpers.basic import BaseTest, assertTensorAlmostEqual


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
    def test_init_single_target(self) -> None:
        model = torch.nn.Sequential(torch.nn.Identity(), torch.nn.Identity())
        target_modules = [model[0]]

        hook_module = output_hook.ModuleOutputsHook(target_modules)
        self.assertEqual(len(hook_module.hooks), len(target_modules))

        n_hooks = _count_forward_hooks(model, "module_outputs_forward_hook")
        self.assertEqual(n_hooks, len(target_modules))

        outputs = dict.fromkeys(target_modules, None)
        self.assertEqual(outputs, hook_module.outputs)
        self.assertEqual(list(hook_module.targets), target_modules)
        self.assertTrue(hook_module.is_ready)

    def test_init_multiple_targets(self) -> None:
        model = torch.nn.Sequential(torch.nn.Identity(), torch.nn.Identity())
        target_modules = [model[0], model[1]]

        hook_module = output_hook.ModuleOutputsHook(target_modules)
        self.assertEqual(len(hook_module.hooks), len(target_modules))

        n_hooks = _count_forward_hooks(model, "module_outputs_forward_hook")
        self.assertEqual(n_hooks, len(target_modules))

        outputs = dict.fromkeys(target_modules, None)
        self.assertEqual(outputs, hook_module.outputs)
        self.assertEqual(list(hook_module.targets), target_modules)
        self.assertTrue(hook_module.is_ready)

    def test_init_hook_duplication_fix(self) -> None:
        model = torch.nn.Sequential(torch.nn.Identity(), torch.nn.Identity())
        for i in range(5):
            _ = output_hook.ModuleOutputsHook([model[1]])
        n_hooks = _count_forward_hooks(model, "module_outputs_forward_hook")
        self.assertEqual(n_hooks, 1)

    def test_init_multiple_targets_remove_hooks(self) -> None:
        model = torch.nn.Sequential(torch.nn.Identity(), torch.nn.Identity())
        target_modules = [model[0], model[1]]

        hook_module = output_hook.ModuleOutputsHook(target_modules)

        n_hooks = _count_forward_hooks(model, "module_outputs_forward_hook")
        self.assertEqual(n_hooks, len(target_modules))

        hook_module.remove_hooks()

        n_hooks = _count_forward_hooks(model, "module_outputs_forward_hook")
        self.assertEqual(n_hooks, 0)

    def test_init_multiple_targets_del(self) -> None:
        model = torch.nn.Sequential(torch.nn.Identity(), torch.nn.Identity())
        target_modules = [model[0], model[1]]

        hook_module = output_hook.ModuleOutputsHook(target_modules)

        n_hooks = _count_forward_hooks(model, "module_outputs_forward_hook")
        self.assertEqual(n_hooks, len(target_modules))

        del hook_module

        n_hooks = _count_forward_hooks(model, "module_outputs_forward_hook")
        self.assertEqual(n_hooks, 0)

    def test_reset_outputs_multiple_targets(self) -> None:
        model = torch.nn.Sequential(torch.nn.Identity(), torch.nn.Identity())
        target_modules = [model[0], model[1]]
        test_input = torch.randn(1, 3, 4, 4)

        hook_module = output_hook.ModuleOutputsHook(target_modules)
        self.assertTrue(hook_module.is_ready)

        _ = model(test_input)

        self.assertFalse(hook_module.is_ready)

        outputs_dict = hook_module.outputs
        i = 0
        for target, activations in outputs_dict.items():
            self.assertEqual(target, target_modules[i])
            assertTensorAlmostEqual(self, activations, test_input)
            i+=1

        hook_module._reset_outputs()

        self.assertTrue(hook_module.is_ready)

        expected_outputs = dict.fromkeys(target_modules, None)
        self.assertEqual(hook_module.outputs, expected_outputs)

    def test_consume_outputs_multiple_targets(self) -> None:
        model = torch.nn.Sequential(torch.nn.Identity(), torch.nn.Identity())
        target_modules = [model[0], model[1]]
        test_input = torch.randn(1, 3, 4, 4)

        hook_module = output_hook.ModuleOutputsHook(target_modules)
        self.assertTrue(hook_module.is_ready)

        _ = model(test_input)

        self.assertFalse(hook_module.is_ready)

        test_outputs_dict = hook_module.outputs
        self.assertIsInstance(test_outputs_dict, dict)
        self.assertEqual(len(test_outputs_dict), len(target_modules))

        i = 0
        for target, activations in test_outputs_dict.items():
            self.assertEqual(target, target_modules[i])
            assertTensorAlmostEqual(self, activations, test_input)
            i+=1

        test_output = hook_module.consume_outputs()

        self.assertTrue(hook_module.is_ready)

        expected_outputs = dict.fromkeys(target_modules, None)
        self.assertEqual(test_output, expected_outputs)
        self.assertEqual(hook_module.outputs, expected_outputs)


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
