#!/usr/bin/env python3
import unittest

import torch

import captum.optim._utils.image.common as common
from tests.helpers.basic import BaseTest, assertTensorAlmostEqual


class TestMakeGridImage(BaseTest):
    def test_make_grid_image_single_tensor(self) -> None:
        """
        Border has a size of 1, and we are using 3 in a row:
        total_w = (img_per_row * border) + border + (img_per_row * tensor_dim_w)
        total_h = (n_column * border) + border + (n_column * tensor_dim_h)
        (3 * 1) + 1 + (3 * 2) = 10
        (2 * 1) + 1 + (2 * 2) = 7
        The n_column value is calculated and used internally by make_grid_image.
        It's calculated based on the given nrow value and how tensors in the
        list / batch dimension there are.
        """
        test_input = torch.ones(6, 1, 2, 2)
        test_output = common.make_grid_image(
            test_input, images_per_row=3, padding=1, pad_value=0.0
        )
        expected_output = torch.tensor(
            [
                [
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0],
                        [0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0],
                        [0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    ]
                ]
            ]
        )
        self.assertEqual(list(expected_output.shape), [1, 1, 7, 10])
        assertTensorAlmostEqual(self, test_output, expected_output, 0)

    def test_make_grid_image_tensor_list(self) -> None:
        test_input = [torch.ones(1, 1, 2, 2) for i in range(6)]
        test_output = common.make_grid_image(
            test_input, images_per_row=3, padding=1, pad_value=0.0
        )
        expected_output = torch.tensor(
            [
                [
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0],
                        [0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0],
                        [0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    ]
                ]
            ]
        )
        self.assertEqual(list(expected_output.shape), [1, 1, 7, 10])
        assertTensorAlmostEqual(self, test_output, expected_output, 0)

    def test_make_grid_image_single_tensor_fewer_tiles(self) -> None:
        test_input = torch.ones(4, 1, 2, 2)
        test_output = common.make_grid_image(
            test_input, images_per_row=3, padding=1, pad_value=0.0
        )
        expected_output = torch.tensor(
            [
                [
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0],
                        [0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    ]
                ]
            ]
        )
        self.assertEqual(list(expected_output.shape), [1, 1, 7, 10])
        assertTensorAlmostEqual(self, test_output, expected_output, 0)

    def test_make_grid_image_single_tensor_padding(self) -> None:
        test_input = torch.ones(4, 1, 2, 2)
        test_output = common.make_grid_image(
            test_input, images_per_row=2, padding=2, pad_value=0.0
        )
        expected_output = torch.tensor(
            [
                [
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    ]
                ]
            ]
        )
        self.assertEqual(list(expected_output.shape), [1, 1, 10, 10])
        assertTensorAlmostEqual(self, test_output, expected_output, 0)

    def test_make_grid_image_single_tensor_pad_value(self) -> None:
        test_input = torch.ones(4, 1, 2, 2)
        test_output = common.make_grid_image(
            test_input, images_per_row=2, padding=1, pad_value=5.0
        )
        expected_output = torch.tensor(
            [
                [
                    [
                        [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
                        [5.0, 1.0, 1.0, 5.0, 1.0, 1.0, 5.0],
                        [5.0, 1.0, 1.0, 5.0, 1.0, 1.0, 5.0],
                        [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
                        [5.0, 1.0, 1.0, 5.0, 1.0, 1.0, 5.0],
                        [5.0, 1.0, 1.0, 5.0, 1.0, 1.0, 5.0],
                        [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
                    ]
                ]
            ]
        )
        self.assertEqual(list(expected_output.shape), [1, 1, 7, 7])
        assertTensorAlmostEqual(self, test_output, expected_output, 0)

    def test_make_grid_image_single_tensor_pad_value_cuda(self) -> None:
        if not torch.cuda.is_available():
            raise unittest.SkipTest(
                "Skipping make_image_grid CUDA test due to not supporting" + " CUDA."
            )
        test_input = torch.ones(4, 1, 2, 2).cuda()
        test_output = common.make_grid_image(
            test_input, images_per_row=2, padding=1, pad_value=5.0
        )
        expected_output = torch.tensor(
            [
                [
                    [
                        [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
                        [5.0, 1.0, 1.0, 5.0, 1.0, 1.0, 5.0],
                        [5.0, 1.0, 1.0, 5.0, 1.0, 1.0, 5.0],
                        [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
                        [5.0, 1.0, 1.0, 5.0, 1.0, 1.0, 5.0],
                        [5.0, 1.0, 1.0, 5.0, 1.0, 1.0, 5.0],
                        [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
                    ]
                ]
            ]
        )
        self.assertEqual(list(expected_output.shape), [1, 1, 7, 7])
        self.assertTrue(test_output.is_cuda)
        assertTensorAlmostEqual(self, test_output, expected_output, 0)

    def test_make_grid_image_single_tensor_pad_value_jit_module(self) -> None:
        if torch.__version__ <= "1.8.0":
            raise unittest.SkipTest(
                "Skipping make_image_grid JIT module test due to"
                + "  insufficient Torch version."
            )
        test_input = torch.ones(4, 1, 2, 2)
        jit_make_grid_image = torch.jit.script(common.make_grid_image)
        test_output = jit_make_grid_image(
            test_input, images_per_row=2, padding=1, pad_value=5.0
        )
        expected_output = torch.tensor(
            [
                [
                    [
                        [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
                        [5.0, 1.0, 1.0, 5.0, 1.0, 1.0, 5.0],
                        [5.0, 1.0, 1.0, 5.0, 1.0, 1.0, 5.0],
                        [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
                        [5.0, 1.0, 1.0, 5.0, 1.0, 1.0, 5.0],
                        [5.0, 1.0, 1.0, 5.0, 1.0, 1.0, 5.0],
                        [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
                    ]
                ]
            ]
        )
        self.assertEqual(list(expected_output.shape), [1, 1, 7, 7])
        assertTensorAlmostEqual(self, test_output, expected_output, 0)


class TestGetNeuronPos(BaseTest):
    def test_get_neuron_pos_hw(self) -> None:
        W, H = 128, 128
        x, y = common.get_neuron_pos(H, W)

        self.assertEqual(x, W // 2)
        self.assertEqual(y, H // 2)

    def test_get_neuron_pos_xy(self) -> None:
        W, H = 128, 128
        x, y = common.get_neuron_pos(H, W, 5, 5)

        self.assertEqual(x, 5)
        self.assertEqual(y, 5)

    def test_get_neuron_pos_x_none(self) -> None:
        W, H = 128, 128
        x, y = common.get_neuron_pos(H, W, 5, None)

        self.assertEqual(x, 5)
        self.assertEqual(y, H // 2)

    def test_get_neuron_pos_none_y(self) -> None:
        W, H = 128, 128
        x, y = common.get_neuron_pos(H, W, None, 5)

        self.assertEqual(x, W // 2)
        self.assertEqual(y, 5)


class TestNChannelsToRGB(BaseTest):
    def test_nchannels_to_rgb_collapse(self) -> None:
        test_input = torch.randn(1, 6, 224, 224)
        test_output = common.nchannels_to_rgb(test_input)
        self.assertEqual(list(test_output.size()), [1, 3, 224, 224])

    def test_nchannels_to_rgb_increase(self) -> None:
        test_input = torch.randn(1, 2, 224, 224)
        test_output = common.nchannels_to_rgb(test_input)
        self.assertEqual(list(test_output.size()), [1, 3, 224, 224])


class TestWeightsToHeatmap2D(BaseTest):
    def test_weights_to_heatmap_2d(self) -> None:
        x = torch.ones(5, 4)
        x[0:1, 0:4] = x[0:1, 0:4] * 0.2
        x[1:2, 0:4] = x[1:2, 0:4] * 0.8
        x[2:3, 0:4] = x[2:3, 0:4] * 0.0
        x[3:4, 0:4] = x[3:4, 0:4] * -0.2
        x[4:5, 0:4] = x[4:5, 0:4] * -0.8

        x_out = common.weights_to_heatmap_2d(x)

        x_out_expected = torch.tensor(
            [
                [
                    [0.9639, 0.9639, 0.9639, 0.9639],
                    [0.8580, 0.8580, 0.8580, 0.8580],
                    [0.9686, 0.9686, 0.9686, 0.9686],
                    [0.8102, 0.8102, 0.8102, 0.8102],
                    [0.2408, 0.2408, 0.2408, 0.2408],
                ],
                [
                    [0.8400, 0.8400, 0.8400, 0.8400],
                    [0.2588, 0.2588, 0.2588, 0.2588],
                    [0.9686, 0.9686, 0.9686, 0.9686],
                    [0.8902, 0.8902, 0.8902, 0.8902],
                    [0.5749, 0.5749, 0.5749, 0.5749],
                ],
                [
                    [0.7851, 0.7851, 0.7851, 0.7851],
                    [0.2792, 0.2792, 0.2792, 0.2792],
                    [0.9686, 0.9686, 0.9686, 0.9686],
                    [0.9294, 0.9294, 0.9294, 0.9294],
                    [0.7624, 0.7624, 0.7624, 0.7624],
                ],
            ]
        )
        assertTensorAlmostEqual(self, x_out, x_out_expected, delta=0.01)

    def test_weights_to_heatmap_2d_cuda(self) -> None:
        if not torch.cuda.is_available():
            raise unittest.SkipTest(
                "Skipping weights_to_heatmap_2d CUDA test due to not supporting CUDA."
            )
        x = torch.ones(5, 4)
        x[0:1, 0:4] = x[0:1, 0:4] * 0.2
        x[1:2, 0:4] = x[1:2, 0:4] * 0.8
        x[2:3, 0:4] = x[2:3, 0:4] * 0.0
        x[3:4, 0:4] = x[3:4, 0:4] * -0.2
        x[4:5, 0:4] = x[4:5, 0:4] * -0.8

        x_out = common.weights_to_heatmap_2d(x.cuda())

        x_out_expected = torch.tensor(
            [
                [
                    [0.9639, 0.9639, 0.9639, 0.9639],
                    [0.8580, 0.8580, 0.8580, 0.8580],
                    [0.9686, 0.9686, 0.9686, 0.9686],
                    [0.8102, 0.8102, 0.8102, 0.8102],
                    [0.2408, 0.2408, 0.2408, 0.2408],
                ],
                [
                    [0.8400, 0.8400, 0.8400, 0.8400],
                    [0.2588, 0.2588, 0.2588, 0.2588],
                    [0.9686, 0.9686, 0.9686, 0.9686],
                    [0.8902, 0.8902, 0.8902, 0.8902],
                    [0.5749, 0.5749, 0.5749, 0.5749],
                ],
                [
                    [0.7851, 0.7851, 0.7851, 0.7851],
                    [0.2792, 0.2792, 0.2792, 0.2792],
                    [0.9686, 0.9686, 0.9686, 0.9686],
                    [0.9294, 0.9294, 0.9294, 0.9294],
                    [0.7624, 0.7624, 0.7624, 0.7624],
                ],
            ]
        )
        assertTensorAlmostEqual(self, x_out, x_out_expected, delta=0.01)
        self.assertTrue(x_out.is_cuda)
