#!/usr/bin/env python3
import unittest

import torch
from captum.optim._models.inception_v1 import googlenet
from tests.helpers.basic import BaseTest


class TestInceptionV1(BaseTest):
    def test_load_and_forward_basic_inceptionv1(self) -> None:
        x = torch.randn(1, 3, 224, 224)
        model = googlenet(pretrained=True)
        try:
            model(x)
            test = True
        except Exception:
            test = False
        assert test

    def test_load_and_forward_diff_sizes_inceptionv1(self) -> None:
        x = torch.randn(1, 3, 512, 512)
        x2 = torch.randn(1, 3, 383, 511)
        model = googlenet(pretrained=True)
        try:
            model(x)
            model(x2)
            test = True
        except Exception:
            test = False
        assert test


if __name__ == "__main__":
    unittest.main()
