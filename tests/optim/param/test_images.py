#!/usr/bin/env python3
import unittest

import numpy as np
import torch
from captum.optim._param.image import images
from tests.helpers.basic import (
    BaseTest,
    assertArraysAlmostEqual,
    assertTensorAlmostEqual,
)
from tests.optim.helpers import numpy_image


class TestFFTImage(BaseTest):
    def test_pytorch_fftfreq(self) -> None:
        assertArraysAlmostEqual(
            images.FFTImage.pytorch_fftfreq(4, 4).numpy(), np.fft.fftfreq(4, 4)
        )

    def test_rfft2d_freqs(self) -> None:
        height = 2
        width = 3
        assertArraysAlmostEqual(
            images.FFTImage.rfft2d_freqs(height, width).numpy(),
            numpy_image.rfft2d_freqs(height, width),
        )


class TestPixelImage(BaseTest):
    def test_pixelimage_random(self) -> None:
        if torch.__version__ == "1.2.0":
            raise unittest.SkipTest(
                "Skipping PixelImage random due to insufficient Torch version."
            )
        size = (224, 224)
        channels = 3
        image_param = images.PixelImage(size=size, channels=channels)

        self.assertEqual(image_param.image.dim(), 4)
        self.assertEqual(image_param.image.size(0), 1)
        self.assertEqual(image_param.image.size(1), channels)
        self.assertEqual(image_param.image.size(2), size[0])
        self.assertEqual(image_param.image.size(3), size[1])

    def test_pixelimage_init(self) -> None:
        if torch.__version__ == "1.2.0":
            raise unittest.SkipTest(
                "Skipping PixelImage init due to insufficient Torch version."
            )
        size = (224, 224)
        channels = 3
        init_tensor = torch.randn(3, 224, 224)
        image_param = images.PixelImage(size=size, channels=channels, init=init_tensor)

        self.assertEqual(image_param.image.dim(), 4)
        self.assertEqual(image_param.image.size(0), 1)
        self.assertEqual(image_param.image.size(1), channels)
        self.assertEqual(image_param.image.size(2), size[0])
        self.assertEqual(image_param.image.size(3), size[1])
        assertTensorAlmostEqual(self, image_param.image, init_tensor, 0)


if __name__ == "__main__":
    unittest.main()
