#!/usr/bin/env python3
import unittest
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from captum.optim._param.image import transform
from tests.helpers.basic import (
    BaseTest,
    assertArraysAlmostEqual,
    assertTensorAlmostEqual,
)


class TestRandSelect(BaseTest):
    def test_rand_select(self) -> None:
        a = (1, 2, 3, 4, 5)
        b = torch.Tensor([0.1, -5, 56.7, 99.0])

        self.assertIn(transform.rand_select(a), a)
        self.assertIn(transform.rand_select(b), b)


class TestRandomScale(BaseTest):
    def test_random_scale(self) -> None:
        scale_module = transform.RandomScale(scale=(1, 0.975, 1.025, 0.95, 1.05))
        test_tensor = torch.ones(1, 3, 3, 3)

        # Test rescaling
        assertTensorAlmostEqual(
            self,
            scale_module.scale_tensor(test_tensor, 0.5),
            torch.ones(3, 1).repeat(3, 1, 3).unsqueeze(0),
            0,
        )

        assertTensorAlmostEqual(
            self,
            scale_module.scale_tensor(test_tensor, 1.5),
            torch.tensor(
                [
                    [0.2500, 0.5000, 0.2500],
                    [0.5000, 1.0000, 0.5000],
                    [0.2500, 0.5000, 0.2500],
                ]
            )
            .repeat(3, 1, 1)
            .unsqueeze(0),
            0,
        )

    def test_random_scale_matrix(self) -> None:
        scale_module = transform.RandomScale(scale=(1, 0.975, 1.025, 0.95, 1.05))
        test_tensor = torch.ones(1, 3, 3, 3)
        # Test scale matrices

        assertTensorAlmostEqual(
            self,
            scale_module.get_scale_mat(0.5, test_tensor.device, test_tensor.dtype),
            torch.tensor([[0.5000, 0.0000, 0.0000], [0.0000, 0.5000, 0.0000]]),
            0,
        )

        assertTensorAlmostEqual(
            self,
            scale_module.get_scale_mat(1.24, test_tensor.device, test_tensor.dtype),
            torch.tensor([[1.2400, 0.0000, 0.0000], [0.0000, 1.2400, 0.0000]]),
            0,
        )


class TestRandomSpatialJitter(BaseTest):
    def test_random_spatial_jitter(self) -> None:

        spatialjitter = transform.RandomSpatialJitter(3)
        test_input = torch.eye(4, 4).repeat(3, 1, 1).unsqueeze(0)

        assertTensorAlmostEqual(
            self,
            spatialjitter.translate_tensor(test_input, torch.tensor([4, 4])),
            torch.tensor(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0, 1.0],
                ]
            )
            .repeat(3, 1, 1)
            .unsqueeze(0),
            0,
        )

        spatialjitter = transform.RandomSpatialJitter(2)

        assertTensorAlmostEqual(
            self,
            spatialjitter.translate_tensor(test_input, torch.tensor([0, 3])),
            torch.tensor(
                [
                    [0.0, 1.0, 0.0, 1.0],
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                ]
            )
            .repeat(3, 1, 1)
            .unsqueeze(0),
            0,
        )

    def numpy_spatialjitter(
        self, x: np.ndarray, pad_range: int, insets: List[int]
    ) -> np.ndarray:
        x = np.pad(x, (pad_range, pad_range), "reflect")
        x = np.roll(x, (pad_range - insets[1]), axis=0)
        x = np.roll(x, (pad_range - insets[0]), axis=1)

        h_crop = x.shape[0] - (pad_range * 2)
        w_crop = x.shape[1] - (pad_range * 2)
        sw, sh = x.shape[1] // 2 - (w_crop // 2), x.shape[0] // 2 - (h_crop // 2)
        x = x[sh : sh + h_crop, sw : sw + w_crop]
        return x

    def test_random_spatial_jitter_numpy(self) -> None:
        pad_range = 3
        test_insets = [2, 3]

        spatial_mod = transform.RandomSpatialJitter(pad_range)
        test_tensor = torch.eye(4, 4).repeat(3, 1, 1).unsqueeze(0)
        test_tensor = spatial_mod.translate_tensor(
            test_tensor, torch.tensor(test_insets)
        ).squeeze(0)

        test_array = self.numpy_spatialjitter(np.eye(4, 4), pad_range, test_insets)

        assertArraysAlmostEqual(test_tensor[0].numpy(), test_array, 0)
        assertArraysAlmostEqual(test_tensor[1].numpy(), test_array, 0)
        assertArraysAlmostEqual(test_tensor[2].numpy(), test_array, 0)


class TestCenterCrop(BaseTest):
    def test_center_crop(self) -> None:
        pad = (1, 1, 1, 1)
        test_tensor = (
            F.pad(F.pad(torch.ones(2, 2), pad=pad), pad=pad, value=1)
            .repeat(3, 1, 1)
            .unsqueeze(0)
        )

        crop_tensor = transform.CenterCrop(size=3)

        assertTensorAlmostEqual(
            self,
            crop_tensor(test_tensor),
            torch.tensor(
                [
                    [1.0, 1.0, 0.0],
                    [1.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0],
                ]
            )
            .repeat(3, 1, 1)
            .unsqueeze(0),
            0,
        )

        crop_tensor = transform.CenterCrop(size=(4, 0))

        assertTensorAlmostEqual(
            self,
            crop_tensor(test_tensor),
            torch.tensor(
                [
                    [1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
                ]
            )
            .repeat(3, 2, 1)
            .unsqueeze(0),
            0,
        )

    def center_crop_numpy(self, input: np.ndarray, crop_val) -> np.ndarray:
        if type(crop_val) is not list and type(crop_val) is not tuple:
            crop_val = [crop_val] * 2
        assert len(crop_val) == 2
        assert input.ndim == 3 or input.ndim == 4
        if input.ndim == 4:
            h, w = input.shape[2], input.shape[3]
        elif input.ndim == 3:
            h, w = input.shape[1], input.shape[2]
        h_crop = h - crop_val[0]
        w_crop = w - crop_val[1]
        sw, sh = w // 2 - (w_crop // 2), h // 2 - (h_crop // 2)
        return input[..., sh : sh + h_crop, sw : sw + w_crop]

    def test_random_crop_numpy(self) -> None:
        crop_vals = (3, 3)
        crop_mod = transform.CenterCrop(crop_vals)

        pad = (1, 1, 1, 1)
        test_tensor = (
            F.pad(F.pad(torch.ones(2, 2), pad=pad), pad=pad, value=1)
            .repeat(3, 1, 1)
            .unsqueeze(0)
        )
        test_tensor_cropped = crop_mod(test_tensor).squeeze(0)

        test_array = np.pad(
            np.pad(np.ones((2, 2)), pad_width=1), pad_width=1, constant_values=(1)
        )[None, None, :]
        test_array = self.center_crop_numpy(test_array, crop_vals)

        assertArraysAlmostEqual(test_tensor_cropped.numpy(), test_array, 0)


class TestBlendAlpha(BaseTest):
    def test_blend_alpha(self) -> None:
        rgb_tensor = torch.ones(3, 3, 3)
        alpha_tensor = ((torch.eye(3, 3) + torch.eye(3, 3).flip(1)) / 2).repeat(1, 1, 1)
        test_tensor = torch.cat([rgb_tensor, alpha_tensor]).unsqueeze(0)

        background_tensor = torch.ones_like(rgb_tensor) * 5
        blend_alpha = transform.BlendAlpha(background=background_tensor)

        assertTensorAlmostEqual(
            self,
            blend_alpha(test_tensor),
            torch.tensor(
                [
                    [3.0, 5.0, 3.0],
                    [5.0, 1.0, 5.0],
                    [3.0, 5.0, 3.0],
                ]
            )
            .repeat(3, 1, 1)
            .unsqueeze(0),
            0,
        )

    def blend_alpha_numpy(
        self, x: np.ndarray, background: Optional[np.ndarray]
    ) -> np.ndarray:
        assert x.shape[1] == 4
        assert x.ndim == 4
        rgb, alpha = x[:, :3, ...], x[:, 3:4, ...]
        background = (
            background if background is not None else np.random.randn(*rgb.shape)
        )
        blended = alpha * rgb + (1 - alpha) * background
        return blended

    def test_blend_alpha_numpy(self) -> None:
        rgb_tensor = torch.ones(3, 3, 3)
        alpha_tensor = ((torch.eye(3, 3) + torch.eye(3, 3).flip(1)) / 2).repeat(1, 1, 1)
        test_tensor = torch.cat([rgb_tensor, alpha_tensor]).unsqueeze(0)

        background_tensor = torch.ones_like(rgb_tensor) * 5
        blend_alpha_pytorch = transform.BlendAlpha(background=background_tensor)
        test_tensor = blend_alpha_pytorch(test_tensor)

        rgb_array = np.ones((3, 3, 3))
        alpha_array = (np.add(np.eye(3, 3), np.flip(np.eye(3, 3), 1)) / 2)[None, :]
        test_array = np.concatenate([rgb_array, alpha_array])[None, :]

        background_array = np.ones(rgb_array.shape) * 5
        test_array = self.blend_alpha_numpy(test_array, background_array)

        assertArraysAlmostEqual(test_tensor.numpy(), test_array, 0)


class TestIgnoreAlpha(BaseTest):
    def test_ignore_alpha(self) -> None:
        ignore_alpha = transform.IgnoreAlpha()
        test_input = torch.ones(1, 4, 3, 3)
        rgb_tensor = ignore_alpha(test_input)
        assert rgb_tensor.size(1) == 3


class TestToRGB(BaseTest):
    def get_matrix(self, matrix: str = "klt") -> torch.Tensor:
        if matrix == "klt":
            KLT = [[0.26, 0.09, 0.02], [0.27, 0.00, -0.05], [0.27, -0.09, 0.03]]
            transform = torch.Tensor(KLT).float()
            transform = transform / torch.max(torch.norm(transform, dim=0))
        else:
            i1i2i3_matrix = [
                [1 / 3, 1 / 3, 1 / 3],
                [1 / 2, 0, -1 / 2],
                [-1 / 4, 1 / 2, -1 / 4],
            ]
            transform = torch.Tensor(i1i2i3_matrix)
        return transform

    def test_to_rgb_i1i2i3(self) -> None:
        to_rgb = transform.ToRGB(transform_name="i1i2i3")
        assertTensorAlmostEqual(self, to_rgb.transform, self.get_matrix("i1i2i3"), 0)

    def test_to_rgb_klt(self) -> None:
        to_rgb = transform.ToRGB(transform_name="klt")
        assertTensorAlmostEqual(self, to_rgb.transform, self.get_matrix("klt"), 0)

    def test_to_rgb(self) -> None:
        if torch.__version__ == "1.2.0":
            raise unittest.SkipTest(
                "Skipping ToRGB forward due to insufficient Torch version."
            )
        to_rgb = transform.ToRGB(transform_name="klt")
        test_tensor = torch.ones(3, 4, 4).unsqueeze(0).refine_names("B", "C", "H", "W")

        rgb_tensor = to_rgb(test_tensor)

        r = torch.ones(4).repeat(4, 1) * 0.8009
        b = torch.ones(4).repeat(4, 1) * 0.4762
        g = torch.ones(4).repeat(4, 1) * 0.4546
        expected_rgb = torch.stack([r, b, g])

        diff_rgb = rgb_tensor.clone() - expected_rgb
        self.assertLessEqual(diff_rgb.max().item(), 4.8340e-05)
        self.assertGreaterEqual(diff_rgb.min().item(), -7.7189e-06)

        inverse_tensor = to_rgb(rgb_tensor.clone(), inverse=True)

        h, w = rgb_tensor.size("H"), rgb_tensor.size("W")
        rgb_flat = rgb_tensor.flatten(("H", "W"), "spatials")
        color_matrix = self.get_matrix("klt")
        rgb_correct = torch.inverse(color_matrix) @ rgb_flat
        rgb_output = rgb_correct.unflatten("spatials", (("H", h), ("W", w)))
        assertTensorAlmostEqual(self, inverse_tensor, rgb_output, 0)

    def test_to_rgb_alpha(self) -> None:
        if torch.__version__ == "1.2.0":
            raise unittest.SkipTest(
                "Skipping ToRGB with Alpha forward due to insufficient Torch version."
            )
        to_rgb = transform.ToRGB(transform_name="klt")
        test_tensor = torch.ones(4, 4, 4).unsqueeze(0).refine_names("B", "C", "H", "W")

        rgb_tensor = to_rgb(test_tensor)

        alpha = torch.ones(4).repeat(4, 1)
        r = torch.ones(4).repeat(4, 1) * 0.8009
        b = torch.ones(4).repeat(4, 1) * 0.4762
        g = torch.ones(4).repeat(4, 1) * 0.4546
        expected_rgb = torch.stack([r, b, g, alpha]).unsqueeze(0)

        diff_rgb = rgb_tensor - expected_rgb
        self.assertLessEqual(diff_rgb.max().item(), 4.8340e-05)
        self.assertGreaterEqual(diff_rgb.min().item(), -7.7189e-06)

        inverse_tensor = to_rgb(rgb_tensor, inverse=True)

        rgb_only, alpha_channel = rgb_tensor[:, :3], rgb_tensor[:, 3:]
        h, w = rgb_only.size("H"), rgb_only.size("W")
        rgb_flat = rgb_only.flatten(("H", "W"), "spatials")
        color_matrix = self.get_matrix("klt")
        rgb_correct = torch.inverse(color_matrix) @ rgb_flat
        rgb_output = rgb_correct.unflatten("spatials", (("H", h), ("W", w)))
        rgb_output = torch.cat([rgb_output, alpha_channel], 1)
        assertTensorAlmostEqual(self, inverse_tensor, rgb_output, 0)


class TestGaussianSmoothing(BaseTest):
    def test_gaussian_smoothing_1d(self) -> None:
        channels = 6
        kernel_size = 3
        sigma = 2.0
        dim = 1
        smoothening_module = transform.GaussianSmoothing(
            channels, kernel_size, sigma, dim
        )

        test_tensor = torch.tensor([1.0, 5.0]).repeat(6, 2).unsqueeze(0)

        diff_tensor = smoothening_module(test_tensor) - torch.tensor(
            [2.4467, 3.5533]
        ).repeat(6, 1).unsqueeze(0)
        self.assertLessEqual(diff_tensor.max().item(), 4.268e-05)
        self.assertGreaterEqual(diff_tensor.min().item(), -4.197e-05)

    def test_gaussian_smoothing_2d(self) -> None:
        channels = 3
        kernel_size = 3
        sigma = 2.0
        dim = 2
        smoothening_module = transform.GaussianSmoothing(
            channels, kernel_size, sigma, dim
        )

        test_tensor = torch.tensor([1.0, 5.0]).repeat(3, 6, 3).unsqueeze(0)

        diff_tensor = smoothening_module(test_tensor) - torch.tensor(
            [2.4467, 3.5533]
        ).repeat(3, 4, 2).unsqueeze(0)
        self.assertLessEqual(diff_tensor.max().item(), 4.5539e-05)
        self.assertGreaterEqual(diff_tensor.min().item(), -4.5539e-05)

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
        t_max = diff_tensor.max().item()
        t_min = diff_tensor.min().item()
        self.assertLessEqual(t_max, 4.8162e-05)
        self.assertGreaterEqual(t_min, 3.3377e-06)


if __name__ == "__main__":
    unittest.main()
