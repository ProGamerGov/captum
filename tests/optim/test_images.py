#!/usr/bin/env python3
import unittest

import torch
import torch.nn.functional as F

from captum.optim._param.image import images
from tests.helpers.basic import BaseTest, assertTensorAlmostEqual


class TestFFTImage(BaseTest):
    def test_pytorch_fftfreq(self) -> None:
        assert torch.all(
            images.FFTImage.pytorch_fftfreq(5).eq(
                torch.tensor([0.0000, 0.2000, 0.4000, -0.4000, -0.2000])
            )
        )
        assert torch.all(
            images.FFTImage.pytorch_fftfreq(4, 4).eq(
                torch.tensor([0.0000, 0.0625, -0.1250, -0.0625])
            )
        )

    def test_rfft2d_freqs(self) -> None:
        assertTensorAlmostEqual(
            self,
            images.FFTImage.rfft2d_freqs(height=2, width=3),
            torch.tensor([[0.0000, 0.3333, 0.3333], [0.5000, 0.6009, 0.6009]]),
        )


if __name__ == "__main__":
    unittest.main()
