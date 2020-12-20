#!/usr/bin/env python3
import unittest

import torch

import captum.optim._utils.circuits as circuits
from captum.optim._models.inception_v1 import googlenet
from tests.helpers.basic import BaseTest, assertTensorAlmostEqual


class TestGetExpandedWeights(BaseTest):
    def test_get_expanded_weights(self) -> None:
        if torch.__version__ == "1.2.0":
            raise unittest.SkipTest(
                "Skipping get_expanded_weights test due to insufficient Torch version."
            )
        model = googlenet(pretrained=True)
        output_tensor = circuits.get_expanded_weights(
            model, model.mixed4c, model.mixed4d
        )
        self.assertTrue(torch.is_tensor(output_tensor))


class TestRemoveConstantPad(BaseTest):
    def test_remove_constant_pad(self) -> None:
        x = torch.randn(1, 4, 5, 5)
        x_padded = torch.nn.functional.pad(a, (4, 4, 4, 4), "constant", value=1.2)
        x_out = circuits.remove_constant_pad(x)
        assertTensorAlmostEqual(self, x_out, x)


if __name__ == "__main__":
    unittest.main()
