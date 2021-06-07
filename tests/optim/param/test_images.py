#!/usr/bin/env python3
import unittest
from typing import List

import numpy as np
import torch

from captum.optim._param.image import images
from captum.optim._param.image.transforms import SymmetricPadding
from tests.helpers.basic import (
    BaseTest,
    assertArraysAlmostEqual,
    assertTensorAlmostEqual,
)
from tests.optim.helpers import numpy_image


class TestImageTensor(BaseTest):
    def test_repr(self) -> None:
        self.assertEqual(str(images.ImageTensor()), "ImageTensor([])")

    def test_new(self) -> None:
        x = torch.ones(5)
        test_tensor = images.ImageTensor(x)
        self.assertTrue(torch.is_tensor(test_tensor))
        self.assertEqual(x.shape, test_tensor.shape)

    def test_new_numpy(self) -> None:
        x = torch.ones(5).numpy()
        test_tensor = images.ImageTensor(x)
        self.assertTrue(torch.is_tensor(test_tensor))
        self.assertEqual(x.shape, test_tensor.shape)

    def test_new_list(self) -> None:
        x = torch.ones(5)
        test_tensor = images.ImageTensor(x.tolist())
        self.assertTrue(torch.is_tensor(test_tensor))
        self.assertEqual(x.shape, test_tensor.shape)

    def test_torch_function(self) -> None:
        x = torch.ones(5)
        image_tensor = images.ImageTensor(x)
        image_tensor = (image_tensor * 1) * torch.ones(5)
        self.assertEqual(image_tensor.sum().item(), torch.ones(5).sum().item())

    def test_load_image_from_url(self) -> None:
        try:
            from PIL import Image  # noqa: F401

        except (ImportError, AssertionError):
            raise unittest.SkipTest(
                "Module Pillow / PIL not found, skipping ImageTensor load from url"
                + " test"
            )
        img_url = (
            "https://github.com/pytorch/captum"
            + "/raw/master/website/static/img/captum_logo.png"
        )
        new_tensor = images.ImageTensor().open(img_url)
        self.assertTrue(torch.is_tensor(new_tensor))
        self.assertEqual(list(new_tensor.shape), [3, 54, 208])

    def test_export_and_open_local_image(self) -> None:
        try:
            from PIL import Image  # noqa: F401

        except (ImportError, AssertionError):
            raise unittest.SkipTest(
                "Module Pillow / PIL not found, skipping ImageTensor export and save"
                + " local image test"
            )
        x = torch.ones(1, 3, 5, 5)
        image_tensor = images.ImageTensor(x)

        filename = "image_tensor.jpg"
        image_tensor.export(filename)
        new_tensor = images.ImageTensor().open(filename)

        self.assertTrue(torch.is_tensor(new_tensor))
        assertTensorAlmostEqual(self, image_tensor, new_tensor)

    def test_natural_image_cuda(self) -> None:
        if not torch.cuda.is_available():
            raise unittest.SkipTest(
                "Skipping ImageTensor CUDA test due to not supporting CUDA."
            )
        image_t = images.ImageTensor().cuda()
        self.assertTrue(image_t.is_cuda)


class TestFFTImage(BaseTest):
    def test_pytorch_fftfreq(self) -> None:
        image = images.FFTImage((1, 1))
        _, _, fftfreq = image.get_fft_funcs()
        assertArraysAlmostEqual(fftfreq(4, 4).numpy(), np.fft.fftfreq(4, 4))

    def test_rfft2d_freqs(self) -> None:
        height = 2
        width = 3
        image = images.FFTImage((1, 1))

        assertTensorAlmostEqual(
            self,
            image.rfft2d_freqs(height, width),
            torch.tensor([[0.0000, 0.3333], [0.5000, 0.6009]]),
        )

    def test_fftimage_forward_randn_init(self) -> None:
        if torch.__version__ <= "1.2.0":
            raise unittest.SkipTest(
                "Skipping FFTImage test due to insufficient Torch version."
            )
        size = (224, 224)

        fftimage = images.FFTImage(size=size)
        fftimage_np = numpy_image.FFTImage(size=size)

        fftimage_tensor = fftimage.forward()
        fftimage_array = fftimage_np.forward()

        self.assertEqual(fftimage_tensor.detach().numpy().shape, fftimage_array.shape)

    def test_fftimage_forward_init_randn_batch(self) -> None:
        if torch.__version__ <= "1.2.0":
            raise unittest.SkipTest(
                "Skipping FFTImage test due to insufficient Torch version."
            )
        size = (224, 224)
        batch = 2

        fftimage = images.FFTImage(size=size, batch=batch)
        fftimage_np = numpy_image.FFTImage(size=size, batch=batch)

        fftimage_tensor = fftimage.forward()
        fftimage_array = fftimage_np.forward()

        self.assertEqual(fftimage_tensor.detach().numpy().shape, fftimage_array.shape)

    def test_fftimage_forward_init_randn_channels(self) -> None:
        if torch.__version__ <= "1.2.0":
            raise unittest.SkipTest(
                "Skipping FFTImage test due to insufficient Torch version."
            )
        size = (224, 224)
        channels = 4

        fftimage = images.FFTImage(size=size, channels=channels)
        fftimage_np = numpy_image.FFTImage(size=size, channels=channels)

        fftimage_tensor = fftimage.forward()
        fftimage_array = fftimage_np.forward()

        self.assertEqual(fftimage_tensor.detach().numpy().shape, fftimage_array.shape)

    def test_fftimage_forward_randn_init_width_odd(self) -> None:
        if torch.__version__ <= "1.2.0":
            raise unittest.SkipTest(
                "Skipping FFTImage test due to insufficient Torch version."
            )
        fftimage = images.FFTImage(size=(512, 405))
        self.assertEqual(list(fftimage.spectrum_scale.shape), [1, 512, 203, 1])
        fftimage_tensor = fftimage().detach()
        self.assertEqual(list(fftimage_tensor.shape), [1, 3, 512, 405])

    def test_fftimage_forward_init_chw(self) -> None:
        if torch.__version__ <= "1.2.0":
            raise unittest.SkipTest(
                "Skipping FFTImage test due to insufficient Torch version."
            )
        size = (224, 224)
        init_tensor = torch.randn(1, 3, 224, 224)
        init_array = init_tensor.numpy()

        fftimage = images.FFTImage(size=size, init=init_tensor)
        fftimage_np = numpy_image.FFTImage(size=size, init=init_array)

        fftimage_tensor = fftimage.forward()
        fftimage_array = fftimage_np.forward()

        self.assertEqual(fftimage_tensor.detach().numpy().shape, fftimage_array.shape)
        assertArraysAlmostEqual(fftimage_tensor.detach().numpy(), fftimage_array, 25.0)

    def test_fftimage_forward_init_bchw(self) -> None:
        if torch.__version__ <= "1.2.0":
            raise unittest.SkipTest(
                "Skipping FFTImage test due to insufficient Torch version."
            )
        size = (224, 224)
        init_tensor = torch.randn(1, 3, 224, 224)
        init_array = init_tensor.numpy()

        fftimage = images.FFTImage(size=size, init=init_tensor)
        fftimage_np = numpy_image.FFTImage(size=size, init=init_array)

        fftimage_tensor = fftimage.forward()
        fftimage_array = fftimage_np.forward()

        self.assertEqual(fftimage_tensor.detach().numpy().shape, fftimage_array.shape)
        assertArraysAlmostEqual(fftimage_tensor.detach().numpy(), fftimage_array, 25.0)

    def test_fftimage_forward_init_batch(self) -> None:
        if torch.__version__ <= "1.2.0":
            raise unittest.SkipTest(
                "Skipping FFTImage test due to insufficient Torch version."
            )
        size = (224, 224)
        batch = 2
        init_tensor = torch.randn(1, 3, 224, 224)
        init_array = init_tensor.numpy()

        fftimage = images.FFTImage(size=size, batch=batch, init=init_tensor)
        fftimage_np = numpy_image.FFTImage(size=size, batch=batch, init=init_array)

        fftimage_tensor = fftimage.forward()
        fftimage_array = fftimage_np.forward()

        self.assertEqual(fftimage_tensor.detach().numpy().shape, fftimage_array.shape)
        assertArraysAlmostEqual(fftimage_tensor.detach().numpy(), fftimage_array, 25.0)


class TestPixelImage(BaseTest):
    def test_pixelimage_random(self) -> None:
        if torch.__version__ <= "1.2.0":
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
        if torch.__version__ <= "1.2.0":
            raise unittest.SkipTest(
                "Skipping PixelImage init due to insufficient Torch version."
            )
        size = (224, 224)
        channels = 3
        init_tensor = torch.randn(channels, *size)
        image_param = images.PixelImage(size=size, channels=channels, init=init_tensor)

        self.assertEqual(image_param.image.dim(), 4)
        self.assertEqual(image_param.image.size(0), 1)
        self.assertEqual(image_param.image.size(1), channels)
        self.assertEqual(image_param.image.size(2), size[0])
        self.assertEqual(image_param.image.size(3), size[1])
        assertTensorAlmostEqual(self, image_param.image, init_tensor, 0)

    def test_pixelimage_init_error(self) -> None:
        if torch.__version__ <= "1.2.0":
            raise unittest.SkipTest(
                "Skipping PixelImage init due to insufficient Torch version."
            )
        size = (224, 224)
        channels = 2
        init_tensor = torch.randn(channels, *size)
        with self.assertRaises(AssertionError):
            images.PixelImage(size=size, channels=channels, init=init_tensor)

    def test_pixelimage_random_forward(self) -> None:
        if torch.__version__ <= "1.2.0":
            raise unittest.SkipTest(
                "Skipping PixelImage random due to insufficient Torch version."
            )
        size = (224, 224)
        channels = 3
        image_param = images.PixelImage(size=size, channels=channels)
        test_tensor = image_param.forward().rename(None)

        self.assertEqual(test_tensor.dim(), 4)
        self.assertEqual(test_tensor.size(0), 1)
        self.assertEqual(test_tensor.size(1), channels)
        self.assertEqual(test_tensor.size(2), size[0])
        self.assertEqual(test_tensor.size(3), size[1])

    def test_pixelimage_init_forward(self) -> None:
        if torch.__version__ <= "1.2.0":
            raise unittest.SkipTest(
                "Skipping PixelImage init due to insufficient Torch version."
            )
        size = (224, 224)
        channels = 3
        init_tensor = torch.randn(3, 224, 224)
        image_param = images.PixelImage(size=size, channels=channels, init=init_tensor)
        test_tensor = image_param.forward().rename(None)

        self.assertEqual(test_tensor.dim(), 4)
        self.assertEqual(test_tensor.size(0), 1)
        self.assertEqual(test_tensor.size(1), channels)
        self.assertEqual(test_tensor.size(2), size[0])
        self.assertEqual(test_tensor.size(3), size[1])
        assertTensorAlmostEqual(self, test_tensor, init_tensor.squeeze(0), 0)


class TestLaplacianImage(BaseTest):
    def test_laplacianimage_random_forward(self) -> None:
        if torch.__version__ <= "1.2.0":
            raise unittest.SkipTest(
                "Skipping LaplacianImage random due to insufficient Torch version."
            )
        size = (224, 224)
        channels = 3
        image_param = images.LaplacianImage(size=size, channels=channels)
        test_tensor = image_param.forward()

        self.assertEqual(test_tensor.dim(), 4)
        self.assertEqual(test_tensor.size(0), 1)
        self.assertEqual(test_tensor.size(1), channels)
        self.assertEqual(test_tensor.size(2), size[0])
        self.assertEqual(test_tensor.size(3), size[1])

    def test_laplacianimage_init(self) -> None:
        if torch.__version__ <= "1.2.0":
            raise unittest.SkipTest(
                "Skipping LaplacianImage random due to insufficient Torch version."
            )
        init_t = torch.zeros(1, 224, 224)
        image_param = images.LaplacianImage(size=(224, 224), channels=3, init=init_t)
        test_np = image_param.forward().detach().numpy()
        assertArraysAlmostEqual(np.ones_like(test_np) * 0.5, test_np)


class TestSharedImage(BaseTest):
    def test_sharedimage_get_offset_single_number(self) -> None:
        if torch.__version__ <= "1.2.0":
            raise unittest.SkipTest(
                "Skipping SharedImage test due to insufficient Torch version."
            )
        shared_shapes = (128 // 2, 128 // 2)
        test_param = lambda: torch.ones(3, 3, 224, 224)  # noqa: E731
        image_param = images.SharedImage(
            shapes=shared_shapes, parameterization=test_param
        )

        offset = image_param.get_offset(4, 3)

        self.assertEqual(len(offset), 3)
        self.assertEqual(offset, [[4, 4, 4, 4]] * 3)

    def test_sharedimage_get_offset_exact(self) -> None:
        if torch.__version__ <= "1.2.0":
            raise unittest.SkipTest(
                "Skipping SharedImage test due to insufficient Torch version."
            )
        shared_shapes = (128 // 2, 128 // 2)
        test_param = lambda: torch.ones(3, 3, 224, 224)  # noqa: E731
        image_param = images.SharedImage(
            shapes=shared_shapes, parameterization=test_param
        )

        offset_vals = ((1, 2, 3, 4), (4, 3, 2, 1), (1, 2, 3, 4))
        offset = image_param.get_offset(offset_vals, 3)

        self.assertEqual(len(offset), 3)
        self.assertEqual(offset, [[int(o) for o in v] for v in offset_vals])

    def test_sharedimage_get_offset_single_set_four_numbers(self) -> None:
        if torch.__version__ <= "1.2.0":
            raise unittest.SkipTest(
                "Skipping SharedImage test due to insufficient Torch version."
            )
        shared_shapes = (128 // 2, 128 // 2)
        test_param = lambda: torch.ones(3, 3, 224, 224)  # noqa: E731
        image_param = images.SharedImage(
            shapes=shared_shapes, parameterization=test_param
        )

        offset_vals = (1, 2, 3, 4)
        offset = image_param.get_offset(offset_vals, 3)

        self.assertEqual(len(offset), 3)
        self.assertEqual(offset, [list(offset_vals)] * 3)

    def test_sharedimage_get_offset_single_set_three_numbers(self) -> None:
        if torch.__version__ <= "1.2.0":
            raise unittest.SkipTest(
                "Skipping SharedImage test due to insufficient Torch version."
            )
        shared_shapes = (128 // 2, 128 // 2)
        test_param = lambda: torch.ones(3, 3, 224, 224)  # noqa: E731
        image_param = images.SharedImage(
            shapes=shared_shapes, parameterization=test_param
        )

        offset_vals = (2, 3, 4)
        offset = image_param.get_offset(offset_vals, 3)

        self.assertEqual(len(offset), 3)
        self.assertEqual(offset, [[0] + list(offset_vals)] * 3)

    def test_sharedimage_get_offset_single_set_two_numbers(self) -> None:
        if torch.__version__ <= "1.2.0":
            raise unittest.SkipTest(
                "Skipping SharedImage test due to insufficient Torch version."
            )
        shared_shapes = (128 // 2, 128 // 2)
        test_param = lambda: torch.ones(3, 3, 224, 224)  # noqa: E731
        image_param = images.SharedImage(
            shapes=shared_shapes, parameterization=test_param
        )

        offset_vals = (3, 4)
        offset = image_param.get_offset(offset_vals, 3)

        self.assertEqual(len(offset), 3)
        self.assertEqual(offset, [[0, 0] + list(offset_vals)] * 3)

    def apply_offset_compare(
        self, x_list: List[torch.Tensor], offset_list: List[List[int]]
    ) -> List[torch.Tensor]:
        A = []
        for x, offset in zip(x_list, offset_list):
            assert x.dim() == 4
            size = list(x.size())

            offset_pad = (
                [[abs(offset[0])] * 2]
                + [[abs(offset[1])] * 2]
                + [[abs(offset[2])] * 2]
                + [[abs(offset[3])] * 2]
            )

            x = SymmetricPadding.apply(x, offset_pad)

            for o, s in zip(offset, range(x.dim())):
                x = torch.roll(x, shifts=o, dims=s)

            x = x[: size[0], : size[1], : size[2], : size[3]]
            A.append(x)
        return A

    def test_apply_offset(self):
        if torch.__version__ <= "1.2.0":
            raise unittest.SkipTest(
                "Skipping SharedImage test due to insufficient Torch version."
            )
        size = (4, 3, 224, 224)
        shared_shapes = (128 // 2, 128 // 2)
        offset_vals = (2, 3, 4, 5)
        test_param = lambda: torch.ones(*size)  # noqa: E731
        image_param = images.SharedImage(
            shapes=shared_shapes, parameterization=test_param, offset=offset_vals
        )

        test_x_list = [torch.ones(*size) for x in range(size[0])]
        output_A = image_param.apply_offset(test_x_list)

        x_list = [torch.ones(*size) for x in range(size[0])]
        self.assertEqual(image_param.offset, [list(offset_vals)])

        offset_list = image_param.offset
        expected_A = self.apply_offset_compare(x_list, offset_list)

        for t_expected, t_output in zip(expected_A, output_A):
            assertTensorAlmostEqual(self, t_expected, t_output)

    def test_interpolate_tensor(self) -> None:
        if torch.__version__ <= "1.2.0":
            raise unittest.SkipTest(
                "Skipping SharedImage test due to insufficient Torch version."
            )
        shared_shapes = (128 // 2, 128 // 2)
        test_param = lambda: torch.ones(3, 3, 224, 224)  # noqa: E731
        image_param = images.SharedImage(
            shapes=shared_shapes, parameterization=test_param
        )

        size = (224, 224)
        channels = 3
        batch = 1

        test_tensor = torch.ones(6, 4, 128, 128)
        output_tensor = image_param.interpolate_tensor(
            test_tensor, batch, channels, size[0], size[1]
        )

        self.assertEqual(output_tensor.dim(), 4)
        self.assertEqual(output_tensor.size(0), batch)
        self.assertEqual(output_tensor.size(1), channels)
        self.assertEqual(output_tensor.size(2), size[0])
        self.assertEqual(output_tensor.size(3), size[1])

    def test_sharedimage_single_shape_hw_forward(self) -> None:
        if torch.__version__ <= "1.2.0":
            raise unittest.SkipTest(
                "Skipping SharedImage test due to insufficient Torch version."
            )

        shared_shapes = (128 // 2, 128 // 2)
        batch = 6
        channels = 3
        size = (224, 224)
        test_param = lambda: torch.ones(batch, channels, size[0], size[1])  # noqa: E731
        image_param = images.SharedImage(
            shapes=shared_shapes, parameterization=test_param
        )
        test_tensor = image_param.forward()

        self.assertIsNone(image_param.offset)
        self.assertEqual(image_param.shared_init[0].dim(), 4)
        self.assertEqual(
            list(image_param.shared_init[0].shape), [1, 1] + list(shared_shapes)
        )
        self.assertEqual(test_tensor.dim(), 4)
        self.assertEqual(test_tensor.size(0), batch)
        self.assertEqual(test_tensor.size(1), channels)
        self.assertEqual(test_tensor.size(2), size[0])
        self.assertEqual(test_tensor.size(3), size[1])

    def test_sharedimage_single_shape_chw_forward(self) -> None:
        if torch.__version__ <= "1.2.0":
            raise unittest.SkipTest(
                "Skipping SharedImage test due to insufficient Torch version."
            )

        shared_shapes = (3, 128 // 2, 128 // 2)
        batch = 6
        channels = 3
        size = (224, 224)
        test_param = lambda: torch.ones(batch, channels, size[0], size[1])  # noqa: E731
        image_param = images.SharedImage(
            shapes=shared_shapes, parameterization=test_param
        )
        test_tensor = image_param.forward()

        self.assertIsNone(image_param.offset)
        self.assertEqual(image_param.shared_init[0].dim(), 4)
        self.assertEqual(
            list(image_param.shared_init[0].shape), [1] + list(shared_shapes)
        )
        self.assertEqual(test_tensor.dim(), 4)
        self.assertEqual(test_tensor.size(0), batch)
        self.assertEqual(test_tensor.size(1), channels)
        self.assertEqual(test_tensor.size(2), size[0])
        self.assertEqual(test_tensor.size(3), size[1])

    def test_sharedimage_single_shape_bchw_forward(self) -> None:
        if torch.__version__ <= "1.2.0":
            raise unittest.SkipTest(
                "Skipping SharedImage test due to insufficient Torch version."
            )

        shared_shapes = (1, 3, 128 // 2, 128 // 2)
        batch = 6
        channels = 3
        size = (224, 224)
        test_param = lambda: torch.ones(batch, channels, size[0], size[1])  # noqa: E731
        image_param = images.SharedImage(
            shapes=shared_shapes, parameterization=test_param
        )
        test_tensor = image_param.forward()

        self.assertIsNone(image_param.offset)
        self.assertEqual(image_param.shared_init[0].dim(), 4)
        self.assertEqual(list(image_param.shared_init[0].shape), list(shared_shapes))
        self.assertEqual(test_tensor.dim(), 4)
        self.assertEqual(test_tensor.size(0), batch)
        self.assertEqual(test_tensor.size(1), channels)
        self.assertEqual(test_tensor.size(2), size[0])
        self.assertEqual(test_tensor.size(3), size[1])

    def test_sharedimage_multiple_shapes_forward(self) -> None:
        if torch.__version__ <= "1.2.0":
            raise unittest.SkipTest(
                "Skipping SharedImage test due to insufficient Torch version."
            )

        shared_shapes = (
            (1, 3, 128 // 2, 128 // 2),
            (1, 3, 128 // 4, 128 // 4),
            (1, 3, 128 // 8, 128 // 8),
            (2, 3, 128 // 8, 128 // 8),
            (1, 3, 128 // 16, 128 // 16),
            (2, 3, 128 // 16, 128 // 16),
        )
        batch = 6
        channels = 3
        size = (224, 224)
        test_param = lambda: torch.ones(batch, channels, size[0], size[1])  # noqa: E731
        image_param = images.SharedImage(
            shapes=shared_shapes, parameterization=test_param
        )
        test_tensor = image_param.forward()

        self.assertIsNone(image_param.offset)
        for i in range(len(shared_shapes)):
            self.assertEqual(image_param.shared_init[i].dim(), 4)
            self.assertEqual(
                list(image_param.shared_init[i].shape), list(shared_shapes[i])
            )
        self.assertEqual(test_tensor.dim(), 4)
        self.assertEqual(test_tensor.size(0), batch)
        self.assertEqual(test_tensor.size(1), channels)
        self.assertEqual(test_tensor.size(2), size[0])
        self.assertEqual(test_tensor.size(3), size[1])

    def test_sharedimage_multiple_shapes_diff_len_forward(self) -> None:
        if torch.__version__ <= "1.2.0":
            raise unittest.SkipTest(
                "Skipping SharedImage test due to insufficient Torch version."
            )

        shared_shapes = (
            (128 // 2, 128 // 2),
            (7, 3, 128 // 4, 128 // 4),
            (3, 128 // 8, 128 // 8),
            (2, 4, 128 // 8, 128 // 8),
            (1, 3, 128 // 16, 128 // 16),
            (2, 2, 128 // 16, 128 // 16),
        )
        batch = 6
        channels = 3
        size = (224, 224)
        test_param = lambda: torch.ones(batch, channels, size[0], size[1])  # noqa: E731
        image_param = images.SharedImage(
            shapes=shared_shapes, parameterization=test_param
        )
        test_tensor = image_param.forward()

        self.assertIsNone(image_param.offset)
        for i in range(len(shared_shapes)):
            self.assertEqual(image_param.shared_init[i].dim(), 4)
            s_shape = list(shared_shapes[i])
            s_shape = ([1] * (4 - len(s_shape))) + list(s_shape)
            self.assertEqual(list(image_param.shared_init[i].shape), s_shape)

        self.assertEqual(test_tensor.dim(), 4)
        self.assertEqual(test_tensor.size(0), batch)
        self.assertEqual(test_tensor.size(1), channels)
        self.assertEqual(test_tensor.size(2), size[0])
        self.assertEqual(test_tensor.size(3), size[1])


class TestNaturalImage(BaseTest):
    def test_natural_image_0(self) -> None:
        if torch.__version__ <= "1.2.0":
            raise unittest.SkipTest(
                "Skipping NaturalImage test due to insufficient Torch version."
            )
        image_param = images.NaturalImage(size=(1, 1))
        image_np = image_param.forward().detach().numpy()
        assertArraysAlmostEqual(image_np, np.ones_like(image_np) * 0.5)

    def test_natural_image_1(self) -> None:
        if torch.__version__ <= "1.2.0":
            raise unittest.SkipTest(
                "Skipping NaturalImage test due to insufficient Torch version."
            )
        image_param = images.NaturalImage(init=torch.ones(3, 1, 1))
        image_np = image_param.forward().detach().numpy()
        assertArraysAlmostEqual(image_np, np.ones_like(image_np))

    def test_natural_image_cuda(self) -> None:
        if not torch.cuda.is_available():
            raise unittest.SkipTest(
                "Skipping NaturalImage CUDA test due to not supporting CUDA."
            )
        image_param = images.NaturalImage().cuda()
        self.assertTrue(image_param().is_cuda)

    def test_natural_image_decorrelation_module_none(self) -> None:
        if torch.__version__ <= "1.3.0":
            raise unittest.SkipTest(
                "Skipping NaturalImage test due to insufficient Torch version."
            )
        image_param = images.NaturalImage(
            init=torch.ones(1, 3, 4, 4), decorrelation_module=None
        )
        image = image_param.forward().detach()
        assertTensorAlmostEqual(self, image, torch.ones_like(image))
