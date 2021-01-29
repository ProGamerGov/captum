#!/usr/bin/env python3
import unittest

import torch

import captum.optim._utils.atlas as atlas
from tests.helpers.basic import BaseTest, assertTensorAlmostEqual


class TestNormalizeGrid(BaseTest):
    def test_normalize_grid(self) -> None:
        if torch.__version__ < "1.7.0":
            raise unittest.SkipTest(
                "Skipping normalize grid test due to insufficient Torch version."
            )
        x = torch.arange(0, 2 * 3 * 3).view(3 * 3, 2).float()

        x_out = atlas.normalize_grid(x)

        x_expected = torch.tensor(
            [
                [0.0000, 0.0000],
                [0.1250, 0.1250],
                [0.2500, 0.2500],
                [0.3750, 0.3750],
                [0.5000, 0.5000],
                [0.6250, 0.6250],
                [0.7500, 0.7500],
                [0.8750, 0.8750],
                [1.0000, 1.0000],
            ]
        )

        assertTensorAlmostEqual(self, x_out, x_expected)


class TestCalcGridIndices(BaseTest):
    def test_calc_grid_indices(self) -> None:
        if torch.__version__ < "1.7.0":
            raise unittest.SkipTest(
                "Skipping grid indices test due to insufficient Torch version."
            )
        x = torch.arange(0, 2 * 3 * 3).view(3 * 3, 2).float()
        x = atlas.normalize_grid(x)
        x_indices = atlas.calc_grid_indices(x, grid_size=(2, 2))

        expected_indices = [
            [torch.tensor([0, 1, 2, 3, 4]), torch.tensor([4])],
            [torch.tensor([4]), torch.tensor([4, 5, 6, 7, 8])],
        ]

        for list1, list2 in zip(x_indices, expected_indices):
            for t1, t2 in zip(list1, list2):
                assertTensorAlmostEqual(self, t1, t2)

    def test_calc_grid_indices_extent(self) -> None:
        if torch.__version__ < "1.7.0":
            raise unittest.SkipTest(
                "Skipping grid indices extent test due to insufficient Torch version."
            )
        x = torch.arange(0, 2 * 5 * 5).view(5 * 5, 2).float()
        x = atlas.normalize_grid(x)
        x_indices = atlas.calc_grid_indices(
            x, grid_size=(1, 1), x_extent=(1.0, 2.0), y_extent=(1.0, 2.0)
        )
        assertTensorAlmostEqual(self, x_indices[0][0], torch.tensor([24]), 0)


class TestExtractGridVectors(BaseTest):
    def test_extract_grid_vectors(self) -> None:
        if torch.__version__ < "1.7.0":
            raise unittest.SkipTest(
                "Skipping extract grid vectors test due to insufficient Torch version."
            )
        x_raw = torch.arange(0, 4 * 3 * 3).view(3 * 3, 4).float()
        x = torch.arange(0, 2 * 3 * 3).view(3 * 3, 2).float()
        grid_size = (2, 2)
        x = atlas.normalize_grid(x)
        x_indices = atlas.calc_grid_indices(x, grid_size=grid_size)

        x_vecs, vec_coords = atlas.extract_grid_vectors(
            x_indices, x_raw, grid_size=grid_size, min_density=2
        )

        expected_vecs = torch.tensor([[8.0, 9.0, 10.0, 11.0], [24.0, 25.0, 26.0, 27.0]])
        expected_coords = [(0, 0, 5), (1, 1, 5)]

        assertTensorAlmostEqual(self, x_vecs, expected_vecs)
        self.assertEqual(vec_coords, expected_coords)


class TestCreateAtlasVectors(BaseTest):
    def test_create_atlas_vectors(self) -> None:
        if torch.__version__ < "1.7.0":
            raise unittest.SkipTest(
                "Skipping create atlas vectors test due to insufficient Torch version."
            )
        x_raw = torch.arange(0, 4 * 3 * 3).view(3 * 3, 4).float()
        x = torch.arange(0, 2 * 3 * 3).view(3 * 3, 2).float()
        x_vecs, vec_coords = atlas.create_atlas_vectors(
            x, x_raw, grid_size=(2, 2), min_density=2, normalize=True
        )

        expected_vecs = torch.tensor([[8.0, 9.0, 10.0, 11.0], [24.0, 25.0, 26.0, 27.0]])
        expected_coords = [(0, 0, 5), (1, 1, 5)]

        assertTensorAlmostEqual(self, x_vecs, expected_vecs)
        self.assertEqual(vec_coords, expected_coords)


class TestCreateAtlas(BaseTest):
    def test_create_atlas(self) -> None:
        img_list = [torch.ones(1, 3, 4, 4)] * 2
        expected_coords = [(0, 0), (1, 1)]
        canvas = atlas.create_atlas(img_list, expected_coords, grid_size=(2, 2))
        assertTensorAlmostEqual(self, canvas, torch.ones_like(canvas))


if __name__ == "__main__":
    unittest.main()
