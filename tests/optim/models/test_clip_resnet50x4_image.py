#!/usr/bin/env python3
import unittest
from typing import Type

import torch

from captum.optim.models import clip_resnet50x4_image
from captum.optim.models._common import RedirectedReluLayer, SkipLayer
from tests.helpers.basic import BaseTest, assertTensorAlmostEqual


def _check_layer_in_model(
    self,
    model: torch.nn.Module,
    layer: Type[torch.nn.Module],
) -> None:
    def check_for_layer_in_model(model, layer) -> bool:
        for name, child in model._modules.items():
            if child is not None:
                if isinstance(child, layer):
                    return True
                if check_for_layer_in_model(child, layer):
                    return True
        return False

    self.assertTrue(check_for_layer_in_model(model, layer))


def _check_layer_not_in_model(
    self, model: torch.nn.Module, layer: Type[torch.nn.Module]
) -> None:
    for name, child in model._modules.items():
        if child is not None:
            self.assertNotIsInstance(child, layer)
            _check_layer_not_in_model(self, child, layer)


class TestCLIPResNet50x4(BaseTest):
    def test_load_clip_resnet50x4_with_redirected_relu(self) -> None:
        if torch.__version__ <= "1.6.0":
            raise unittest.SkipTest(
                "Skipping load pretrained CLIP ResNet 50x4 due to insufficient"
                + " Torch version."
            )
        model = clip_resnet50x4_image(
            pretrained=True, replace_relus_with_redirectedrelu=True
        )
        _check_layer_in_model(self, model, RedirectedReluLayer)

    def test_load_clip_resnet50x4_no_redirected_relu(self) -> None:
        if torch.__version__ <= "1.6.0":
            raise unittest.SkipTest(
                "Skipping load pretrained CLIP ResNet 50x4 RedirectedRelu test"
                + " due to insufficient Torch version."
            )
        model = clip_resnet50x4_image(
            pretrained=True, replace_relus_with_redirectedrelu=False
        )
        _check_layer_not_in_model(self, model, RedirectedReluLayer)
        _check_layer_in_model(self, model, torch.nn.ReLU)

    def test_load_clip_resnet50x4_linear(self) -> None:
        if torch.__version__ <= "1.6.0":
            raise unittest.SkipTest(
                "Skipping load pretrained CLIP ResNet 50x4 linear test due to"
                + " insufficient Torch version."
            )
        model = clip_resnet50x4_image(pretrained=True, use_linear_modules_only=True)
        _check_layer_not_in_model(self, model, RedirectedReluLayer)
        _check_layer_not_in_model(self, model, torch.nn.ReLU)
        _check_layer_in_model(self, model, SkipLayer)

    def test_clip_resnet50x4_transform(self) -> None:
        if torch.__version__ <= "1.6.0":
            raise unittest.SkipTest(
                "Skipping CLIP ResNet 50x4 internal transform test due to"
                + " insufficient Torch version."
            )
        x = torch.randn(1, 3, 288, 288).clamp(0, 1)
        model = clip_resnet50x4_image(pretrained=True)
        output = model._transform_input(x)
        expected_output = x.clone() - torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
        expected_output = expected_output / torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
        assertTensorAlmostEqual(self, output, expected_output, 0)

    def test_clip_resnet50x4_transform_warning(self) -> None:
        if torch.__version__ <= "1.6.0":
            raise unittest.SkipTest(
                "Skipping CLIP ResNet 50x4 internal transform warning test due"
                + " to insufficient Torch version."
            )
        x = torch.stack(
            [torch.ones(3, 112, 112) * -1, torch.ones(3, 112, 112) * 2], dim=0
        )
        model = clip_resnet50x4_image(pretrained=True)
        with self.assertWarns(UserWarning):
            model._transform_input(x)

    def test_clip_resnet50x4_load_and_forward(self) -> None:
        if torch.__version__ <= "1.6.0":
            raise unittest.SkipTest(
                "Skipping basic pretrained CLIP ResNet 50x4 forward test due to"
                + " insufficient Torch version."
            )
        x = torch.zeros(1, 3, 288, 288)
        model = clip_resnet50x4_image(pretrained=True)
        output = model(x)
        self.assertEqual([list(output.shape), [1, 640])

    def test_untrained_clip_resnet50x4_load_and_forward(self) -> None:
        if torch.__version__ <= "1.6.0":
            raise unittest.SkipTest(
                "Skipping basic untrained CLIP ResNet 50x4 forward test due to"
                + " insufficient Torch version."
            )
        x = torch.zeros(1, 3, 288, 288)
        model = clip_resnet50x4_image(pretrained=False)
        output = model(x)
        self.assertEqual([list(output.shape), [1, 640])

    def test_clip_resnet50x4_load_and_forward_diff_sizes(self) -> None:
        if torch.__version__ <= "1.6.0":
            raise unittest.SkipTest(
                "Skipping pretrained CLIP ResNet 50x4 forward with different"
                + " sized inputs test due to insufficient Torch version."
            )
        x = torch.zeros(1, 3, 512, 512)
        x2 = torch.zeros(1, 3, 126, 224)
        model = clip_resnet50x4_image(pretrained=True)

        output = model(x)
        output2 = model(x2)

        self.assertEqual([list(output.shape), [1, 640])
        self.assertEqual([list(output2.shape), [1, 640])

    def test_clip_resnet50x4_forward_cuda(self) -> None:
        if torch.__version__ <= "1.6.0":
            raise unittest.SkipTest(
                "Skipping pretrained CLIP ResNet 50x4 forward CUDA test due to"
                + " insufficient Torch version."
            )
        if not torch.cuda.is_available():
            raise unittest.SkipTest(
                "Skipping pretrained CLIP ResNet 50x4 forward CUDA test due to"
                + " not supporting CUDA."
            )
        x = torch.zeros(1, 3, 224, 224).cuda()
        model = clip_resnet50x4_image(pretrained=True).cuda()
        output = model(x)

        self.assertTrue(output.is_cuda)
        self.assertEqual([list(output.shape), [1, 640])

    def test_clip_resnet50x4_jit_module_no_redirected_relu(self) -> None:
        if torch.__version__ <= "1.8.0":
            raise unittest.SkipTest(
                "Skipping pretrained CLIP ResNet 50x4 load & JIT module with no"
                + " redirected relu test due to insufficient Torch version."
            )
        x = torch.zeros(1, 3, 224, 224)
        model = clip_resnet50x4_image(
            pretrained=True, replace_relus_with_redirectedrelu=False
        )
        jit_model = torch.jit.script(model)
        output = jit_model(x)
        self.assertEqual([list(output.shape), [1, 640])
