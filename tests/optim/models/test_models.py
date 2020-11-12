#!/usr/bin/env python3
import unittest

import torch
from captum.optim._models.inception_v1 import googlenet


class TestInceptionV1(BaseTest):
    def test_inceptionv1(self) -> None:
        x = torch.randn(1, 3, 224, 224)
        model = googlenet(pretrained=True)
        try:
            output = model(x)
            test = True
        except:
            test = False
        assert test == True
