#!/usr/bin/env python3
import unittest

import torch
import torch.nn.functional as F

import captum.optim._models.model_utils as model_utils
from captum.optim._models.inception_v1 import googlenet
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


class TestGetLayers(BaseTest):
    def test_get_layers_pretrained_inceptionv1(self) -> None:
        expected_list = [
            "conv1",
            "conv1_relu",
            "pool1",
            "localresponsenorm1",
            "conv2",
            "conv2_relu",
            "conv3",
            "conv3_relu",
            "localresponsenorm2",
            "pool2",
            "mixed3a",
            "mixed3a.conv_1x1",
            "mixed3a.conv_1x1_relu",
            "mixed3a.conv_3x3_reduce",
            "mixed3a.conv_3x3_reduce_relu",
            "mixed3a.conv_3x3",
            "mixed3a.conv_3x3_relu",
            "mixed3a.conv_5x5_reduce",
            "mixed3a.conv_5x5_reduce_relu",
            "mixed3a.conv_5x5",
            "mixed3a.conv_5x5_relu",
            "mixed3a.pool",
            "mixed3a.pool_proj",
            "mixed3a.pool_proj_relu",
            "mixed3b",
            "mixed3b.conv_1x1",
            "mixed3b.conv_1x1_relu",
            "mixed3b.conv_3x3_reduce",
            "mixed3b.conv_3x3_reduce_relu",
            "mixed3b.conv_3x3",
            "mixed3b.conv_3x3_relu",
            "mixed3b.conv_5x5_reduce",
            "mixed3b.conv_5x5_reduce_relu",
            "mixed3b.conv_5x5",
            "mixed3b.conv_5x5_relu",
            "mixed3b.pool",
            "mixed3b.pool_proj",
            "mixed3b.pool_proj_relu",
            "pool3",
            "mixed4a",
            "mixed4a.conv_1x1",
            "mixed4a.conv_1x1_relu",
            "mixed4a.conv_3x3_reduce",
            "mixed4a.conv_3x3_reduce_relu",
            "mixed4a.conv_3x3",
            "mixed4a.conv_3x3_relu",
            "mixed4a.conv_5x5_reduce",
            "mixed4a.conv_5x5_reduce_relu",
            "mixed4a.conv_5x5",
            "mixed4a.conv_5x5_relu",
            "mixed4a.pool",
            "mixed4a.pool_proj",
            "mixed4a.pool_proj_relu",
            "mixed4b",
            "mixed4b.conv_1x1",
            "mixed4b.conv_1x1_relu",
            "mixed4b.conv_3x3_reduce",
            "mixed4b.conv_3x3_reduce_relu",
            "mixed4b.conv_3x3",
            "mixed4b.conv_3x3_relu",
            "mixed4b.conv_5x5_reduce",
            "mixed4b.conv_5x5_reduce_relu",
            "mixed4b.conv_5x5",
            "mixed4b.conv_5x5_relu",
            "mixed4b.pool",
            "mixed4b.pool_proj",
            "mixed4b.pool_proj_relu",
            "mixed4c",
            "mixed4c.conv_1x1",
            "mixed4c.conv_1x1_relu",
            "mixed4c.conv_3x3_reduce",
            "mixed4c.conv_3x3_reduce_relu",
            "mixed4c.conv_3x3",
            "mixed4c.conv_3x3_relu",
            "mixed4c.conv_5x5_reduce",
            "mixed4c.conv_5x5_reduce_relu",
            "mixed4c.conv_5x5",
            "mixed4c.conv_5x5_relu",
            "mixed4c.pool",
            "mixed4c.pool_proj",
            "mixed4c.pool_proj_relu",
            "mixed4d",
            "mixed4d.conv_1x1",
            "mixed4d.conv_1x1_relu",
            "mixed4d.conv_3x3_reduce",
            "mixed4d.conv_3x3_reduce_relu",
            "mixed4d.conv_3x3",
            "mixed4d.conv_3x3_relu",
            "mixed4d.conv_5x5_reduce",
            "mixed4d.conv_5x5_reduce_relu",
            "mixed4d.conv_5x5",
            "mixed4d.conv_5x5_relu",
            "mixed4d.pool",
            "mixed4d.pool_proj",
            "mixed4d.pool_proj_relu",
            "mixed4e",
            "mixed4e.conv_1x1",
            "mixed4e.conv_1x1_relu",
            "mixed4e.conv_3x3_reduce",
            "mixed4e.conv_3x3_reduce_relu",
            "mixed4e.conv_3x3",
            "mixed4e.conv_3x3_relu",
            "mixed4e.conv_5x5_reduce",
            "mixed4e.conv_5x5_reduce_relu",
            "mixed4e.conv_5x5",
            "mixed4e.conv_5x5_relu",
            "mixed4e.pool",
            "mixed4e.pool_proj",
            "mixed4e.pool_proj_relu",
            "pool4",
            "mixed5a",
            "mixed5a.conv_1x1",
            "mixed5a.conv_1x1_relu",
            "mixed5a.conv_3x3_reduce",
            "mixed5a.conv_3x3_reduce_relu",
            "mixed5a.conv_3x3",
            "mixed5a.conv_3x3_relu",
            "mixed5a.conv_5x5_reduce",
            "mixed5a.conv_5x5_reduce_relu",
            "mixed5a.conv_5x5",
            "mixed5a.conv_5x5_relu",
            "mixed5a.pool",
            "mixed5a.pool_proj",
            "mixed5a.pool_proj_relu",
            "mixed5b",
            "mixed5b.conv_1x1",
            "mixed5b.conv_1x1_relu",
            "mixed5b.conv_3x3_reduce",
            "mixed5b.conv_3x3_reduce_relu",
            "mixed5b.conv_3x3",
            "mixed5b.conv_3x3_relu",
            "mixed5b.conv_5x5_reduce",
            "mixed5b.conv_5x5_reduce_relu",
            "mixed5b.conv_5x5",
            "mixed5b.conv_5x5_relu",
            "mixed5b.pool",
            "mixed5b.pool_proj",
            "mixed5b.pool_proj_relu",
            "avgpool",
            "drop",
            "fc",
        ]
        model = googlenet(pretrained=True)
        collected_layers = model_utils.get_model_layers(model)
        assert collected_layers == expected_list


if __name__ == "__main__":
    unittest.main()