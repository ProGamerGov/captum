#!/usr/bin/env python3
import unittest

import torch

from captum.optim.models import clip_resnet50x4_text
from tests.helpers.basic import BaseTest


class TestCLIPResNet50x4(BaseTest):
    def test_clip_resnet50x4_load_and_forward(self) -> None:
        if torch.__version__ <= "1.6.0":
            raise unittest.SkipTest(
                "Skipping basic pretrained CLIP ResNet 50x4 forward test due to"
                + " insufficient Torch version."
            )
        x = torch.cat([torch.tensor([49405, 49406]), torch.zeros(77 - 2)]).int()
        model = clip_resnet50x4_text(pretrained=True)
        output = model(x)
        self.assertEqual(list(output.shape), [1, 640])

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
        x = torch.cat([torch.tensor([49405, 49406]), torch.zeros(77 - 2)]).int().cuda()
        model = clip_resnet50x4_text(pretrained=True).cuda()
        output = model(x)

        self.assertTrue(output.is_cuda)
        self.assertEqual(list(output.shape), [1, 640])

    def test_clip_resnet50x4_jit_module(self) -> None:
        if torch.__version__ <= "1.8.0":
            raise unittest.SkipTest(
                "Skipping pretrained CLIP ResNet 50x4 load & JIT module"
                + " test due to insufficient Torch version."
            )
        x = torch.cat([torch.tensor([49405, 49406]), torch.zeros(77 - 2)]).int()
        model = clip_resnet50x4_text(pretrained=True)
        jit_model = torch.jit.script(model)
        output = jit_model(x)
        self.assertEqual(list(output.shape), [1, 640])
