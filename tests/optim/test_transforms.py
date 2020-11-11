#!/usr/bin/env python3
import unittest

import torch
import torch.nn.functional as F

from captum.optim._param.image import transform
from tests.helpers.basic import BaseTest


class TestRandSelect(BaseTest):
    def test_rand_select(self) -> None:
        a = (1, 2, 3, 4, 5)
        b = torch.Tensor([0.1, -5, 56.7, 99.0])

        assert transform.rand_select(a) in a
        assert transform.rand_select(b) in b


class TestRandomScale(BaseTest):
    def test_random_scale(self) -> None:
        scale_module = transform.RandomScale(scale=(1, 0.975, 1.025, 0.95, 1.05))
        test_tensor = torch.ones(1, 3, 3, 3)

        # Test rescaling
        assert torch.all(
            scale_module.scale_tensor(test_tensor, 0.5).eq(
                torch.ones(3, 1).repeat(3, 1, 3).unsqueeze(0)
            )
        )
        assert torch.all(
            scale_module.scale_tensor(test_tensor, 1.5).eq(
                torch.tensor(
                    [
                        [0.2500, 0.5000, 0.2500],
                        [0.5000, 1.0000, 0.5000],
                        [0.2500, 0.5000, 0.2500],
                    ]
                )
                .repeat(3, 1, 1)
                .unsqueeze(0)
            )
        )

    def test_random_scale_matrix(self) -> None:
        scale_module = transform.RandomScale(scale=(1, 0.975, 1.025, 0.95, 1.05))
        test_tensor = torch.ones(1, 3, 3, 3)
        # Test scale matrices
        assert torch.all(
            scale_module.get_scale_mat(0.5, test_tensor.device, test_tensor.dtype).eq(
                torch.tensor([[0.5000, 0.0000, 0.0000], [0.0000, 0.5000, 0.0000]])
            )
        )
        assert torch.all(
            scale_module.get_scale_mat(1.24, test_tensor.device, test_tensor.dtype).eq(
                torch.tensor([[1.2400, 0.0000, 0.0000], [0.0000, 1.2400, 0.0000]])
            )
        )


class TestRandomSpatialJitter(BaseTest):
    def test_random_spatial_jitter(self) -> None:

        spatialjitter = transform.RandomSpatialJitter(3)
        test_input = torch.eye(4, 4).repeat(3, 1, 1).unsqueeze(0)

        assert torch.all(
            spatialjitter.translate_tensor(test_input, [4, 4]).eq(
                torch.tensor(
                    [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 1.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 1.0, 0.0, 1.0],
                    ]
                )
                .repeat(3, 1, 1)
                .unsqueeze(0)
            )
        )

        spatialjitter = transform.RandomSpatialJitter(2)

        assert torch.all(
            spatialjitter.translate_tensor(test_input, [0, 3]).eq(
                torch.tensor(
                    [
                        [0.0, 1.0, 0.0, 1.0],
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0, 0.0],
                    ]
                )
                .repeat(3, 1, 1)
                .unsqueeze(0)
            )
        )


class TestCenterCrop(BaseTest):
    def test_center_crop(self) -> None:
        pad = (1, 1, 1, 1)
        test_tensor = (
            F.pad(F.pad(torch.ones(2, 2), pad=pad), pad=pad, value=1)
            .repeat(3, 1, 1)
            .unsqueeze(0)
        )

        crop_tensor = transform.CenterCrop(size=3)

        assert torch.all(
            crop_tensor(test_tensor).eq(
                torch.tensor(
                    [
                        [1.0, 1.0, 0.0],
                        [1.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0],
                    ]
                )
                .repeat(3, 1, 1)
                .unsqueeze(0)
            )
        )

        crop_tensor = transform.CenterCrop(size=(4, 0))

        assert torch.all(
            crop_tensor(test_tensor).eq(
                torch.tensor(
                    [
                        [1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
                    ]
                )
                .repeat(3, 2, 1)
                .unsqueeze(0)
            )
        )


class TestBlendAlpha(BaseTest):
    def test_blend_alpha(self) -> None:
        rgb_tensor = torch.ones(3, 3, 3)
        alpha_tensor = ((torch.eye(3, 3) + torch.eye(3, 3).flip(1)) / 2).repeat(1, 1, 1)
        test_tensor = torch.cat([rgb_tensor, alpha_tensor]).unsqueeze(0)

        background_tensor = torch.ones_like(rgb_tensor) * 5
        blend_alpha = transform.BlendAlpha(background=background_tensor)

        assert torch.all(
            blend_alpha(test_tensor).eq(
                torch.tensor(
                    [
                        [3.0, 5.0, 3.0],
                        [5.0, 1.0, 5.0],
                        [3.0, 5.0, 3.0],
                    ]
                )
                .repeat(3, 1, 1)
                .unsqueeze(0)
            )
        )


class TestIgnoreAlpha(BaseTest):
    def test_ignore_alpha(self) -> None:
        ignore_alpha = transform.IgnoreAlpha()
        test_input = torch.ones(1, 4, 3, 3)
        rgb_tensor = ignore_alpha(test_input)
        assert rgb_tensor.size(1) == 3


class TestToRGB(BaseTest):
    def test_to_rgb_i1i2i3(self) -> None:
        to_rgb = transform.ToRGB(transform_name="i1i2i3")
        assert torch.all(
            to_rgb.transform.eq(
                torch.tensor(
                    [
                        [1 / 3, 1 / 3, 1 / 3],
                        [1 / 2, 0, -1 / 2],
                        [-1 / 4, 1 / 2, -1 / 4],
                    ]
                )
            )
        )

    def test_to_rgb_klt(self) -> None:
        to_rgb = transform.ToRGB(transform_name="klt")

        klt_expected = torch.Tensor(
            [[0.26, 0.09, 0.02], [0.27, 0.00, -0.05], [0.27, -0.09, 0.03]]
        )
        klt_expected = klt_expected / torch.max(torch.norm(klt_expected, dim=0))
        assert torch.all(to_rgb.transform.eq(klt_expected))

    def test_to_rgb(self) -> None:
        if torch.__version__ == "1.2.0":
            raise unittest.SkipTest(
                "Skipping ToRGB forward due to insufficient Torch version."
            )
        to_rgb = transform.ToRGB(transform_name="klt")
        test_tensor = torch.ones(3, 4, 4).refine_names("C", "H", "W")

        rgb_tensor = to_rgb(test_tensor)

        r = torch.ones(4).repeat(4, 1) * 0.8009
        b = torch.ones(4).repeat(4, 1) * 0.4762
        g = torch.ones(4).repeat(4, 1) * 0.4546
        expected_rgb = torch.stack([r, b, g])

        diff_rgb = rgb_tensor - expected_rgb
        assert diff_rgb.max() < 4.8340e-05 and diff_rgb.min() > -7.7189e-06

        inverse_tensor = to_rgb(rgb_tensor, inverse=True)

        r_i = torch.ones(4).repeat(4, 1) * 0.9948
        b_i = torch.ones(4).repeat(4, 1) * 0.0675
        g_i = torch.ones(4).repeat(4, 1) * 0.0127
        expected_inverse = torch.stack([r_i, b_i, g_i])

        diff_inverse = inverse_tensor - expected_inverse
        assert diff_inverse.max() < 4.5310e-05 and diff_inverse.min() > -4.7711e-05

    def test_to_rgb_alpha(self) -> None:
        if torch.__version__ == "1.2.0":
            raise unittest.SkipTest(
                "Skipping ToRGB with Alpha forward due to insufficient Torch version."
            )
        to_rgb = transform.ToRGB(transform_name="klt")
        test_tensor = torch.ones(4, 4, 4).refine_names("C", "H", "W")
        alpha = torch.ones(4).repeat(4, 1)

        rgb_tensor = to_rgb(test_tensor)

        r = torch.ones(4).repeat(4, 1) * 0.8009
        b = torch.ones(4).repeat(4, 1) * 0.4762
        g = torch.ones(4).repeat(4, 1) * 0.4546
        expected_rgb = torch.stack([r, b, g, alpha])

        diff_rgb = rgb_tensor - expected_rgb
        assert diff_rgb.max() < 4.8340e-05 and diff_rgb.min() > -7.7189e-06

        inverse_tensor = to_rgb(rgb_tensor, inverse=True)

        r_i = torch.ones(4).repeat(4, 1) * 0.9948
        b_i = torch.ones(4).repeat(4, 1) * 0.0675
        g_i = torch.ones(4).repeat(4, 1) * 0.0127
        expected_inverse = torch.stack([r_i, b_i, g_i, alpha])

        diff_inverse = inverse_tensor - expected_inverse
        assert diff_inverse.max() < 4.5310e-05 and diff_inverse.min() > -4.7711e-05


class TestGaussianSmoothing(BaseTest):
    def test_gaussian_smoothing_1d(self) -> None:
        channels = 6
        kernel_size = 3
        sigma = 2
        dim = 1
        smoothening_module = transform.GaussianSmoothing(
            channels, kernel_size, sigma, dim
        )

        test_tensor = torch.tensor([1.0, 5.0]).repeat(6, 2).unsqueeze(0)

        diff_tensor = smoothening_module(test_tensor) - torch.tensor(
            [2.4467, 3.5533]
        ).repeat(6, 1).unsqueeze(0)
        assert diff_tensor.max() < 4.268e-05 and diff_tensor.min() > -4.197e-05

    def test_gaussian_smoothing_2d(self) -> None:
        channels = 3
        kernel_size = 3
        sigma = 2
        dim = 2
        smoothening_module = transform.GaussianSmoothing(
            channels, kernel_size, sigma, dim
        )

        test_tensor = torch.tensor([1.0, 5.0]).repeat(3, 6, 3).unsqueeze(0)

        diff_tensor = smoothening_module(test_tensor) - torch.tensor(
            [2.4467, 3.5533]
        ).repeat(3, 4, 2).unsqueeze(0)
        assert diff_tensor.max() < 4.5539e-05 and diff_tensor.min() > -4.5539e-05

    def test_gaussian_smoothing_3d(self) -> None:
        channels = 4
        kernel_size = 3
        sigma = 1.021
        dim = 3
        smoothening_module = transform.GaussianSmoothing(
            channels, kernel_size, sigma, dim
        )

        test_tensor = torch.tensor([1.0, 5.0, 1.0]).repeat(4, 6, 6, 2).unsqueeze(0)

        diff_tensor = smoothening_module(test_tensor) - torch.tensor(
            [2.7873, 2.1063, 2.1063, 2.7873]
        ).repeat(4, 4, 4, 1).unsqueeze(0)
        assert (
            diff_tensor.max().item() < 4.8162e-05
            and diff_tensor.min().item() > 3.5762e-06
        )


if __name__ == "__main__":
    unittest.main()
