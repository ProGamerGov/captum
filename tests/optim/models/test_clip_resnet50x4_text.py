#!/usr/bin/env python3
import unittest

import torch

from captum.optim.models import clip_resnet50x4_text
from tests.helpers.basic import BaseTest


class TestCLIPResNet50x4(BaseTest):
    def test_load_clip_resnet50x4(self) -> None:
        if torch.__version__ <= "1.6.0":
            raise unittest.SkipTest(
                "Skipping load pretrained CLIP ResNet 50x4 due to insufficient"
                + " Torch version."
            )
        model = clip_resnet50x4_text(pretrained=True)

    def test_clip_resnet50x4_load_and_forward(self) -> None:
        if torch.__version__ <= "1.6.0":
            raise unittest.SkipTest(
                "Skipping basic pretrained CLIP ResNet 50x4 forward test due to"
                + " insufficient Torch version."
            )
        x = torch.cat([torch.tensor([49405, 49406]), torch.zeros(77-2)])
        model = clip_resnet50x4_text(pretrained=True)
        output = model(x)
        self.assertEqual([list(output.shape), [1, 640])
