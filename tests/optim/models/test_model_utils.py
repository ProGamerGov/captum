#!/usr/bin/env python3
import unittest

import torch
import torch.nn.functional as F

import captum.optim._models.model_utils as model_utils
from tests.helpers.basic import BaseTest


class TestReluLayer(BaseTest):
    def test_relu_layer(self) -> None:
        x = torch.randn(1, 3, 4, 4)

        relu_layer = model_utils.ReluLayer()
        assert torch.all(relu_layer(x).eq(F.relu(x)))


if __name__ == "__main__":
    unittest.main()
