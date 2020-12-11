#!/usr/bin/env python3
import unittest

import torch

import captum.optim._utils.circuits as circuits
from captum.optim._models.inception_v1 import googlenet
from tests.helpers.basic import BaseTest


class TestReplaceLayers(BaseTest):
    def test_activation_catcher(self) -> None:
        model = googlenet(pretrained=True)
        try:
            catch_activ = circuits.ActivationCatcher(targets=[model.mixed4d])
            activ_out = catch_activ(model, torch.zeros(1, 3, 224, 224))
            test = True
        except Exception:
            test = False
        self.assertTrue(test)


class TestMax2AvgPool(BaseTest):
    def test_max2avg_pool(self) -> None:
        model = googlenet(pretrained=True)
        circuits.max2avg_pool(model)
        check_is_not_instance(self, model, torch.nn.MaxPool2d)


class TestGetExpandedWeights(BaseTest):
    def test_get_expanded_weights(self) -> None:
        model = googlenet(pretrained=True)
        try:
            output_tensor = circuits.get_expanded_weights(
                model, model.mixed4c, model.mixed4d
            )
            test = True
        except Exception:
            test = False
        self.assertTrue(test)


def check_is_not_instance(self, model, layer) -> None:
    for name, child in model._modules.items():
        if child is not None:
            self.assertNotIsInstance(child, layer)
            check_is_not_instance(self, child, layer)
