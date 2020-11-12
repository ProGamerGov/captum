#!/usr/bin/env python3
import unittest

import torch
import torch.nn.functional as F

import captum.optim._models.model_utils as model_utils
from tests.helpers.basic import BaseTest


class TestLocalResponseNormLayer(BaseTest):
    def test_local_response_norm_layer(self) -> None:
        size = 5
        alpha = 9.999999747378752e-05
        beta = 0.75
        k = 1

        x = torch.randn(32, 5, 24, 24)
        lrn_layer = model_utils.LocalResponseNormLayer(
            size=size, alpha=alpha, beta=beta, k=k
        )

        assert torch.all(
            lrn_layer(x).eq(
                F.local_response_norm(x, size=size, alpha=alpha, beta=beta, k=k)
            )
        )


class TestReluLayer(BaseTest):
    def test_relu_layer(self) -> None:
        x = torch.randn(1, 3, 4, 4)

        relu_layer = model_utils.ReluLayer()
        assert torch.all(relu_layer(x).eq(F.relu(x)))


class TestRedirectedReluLayer(BaseTest):
    def test_forward_redirected_relu_layer(self) -> None:
        x = torch.randn(1, 3, 4, 4)
        layer = model_utils.RedirectedReluLayer()
        assert torch.all(layer(x).eq(F.relu(x)))


if __name__ == "__main__":
    unittest.main()
