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
        xy_grid = torch.arange(0, 2 * 3 * 3).view(3 * 3, 2).float()

        xy_grid = atlas.normalize_grid(xy_grid)

        xy_grid_expected = torch.tensor(
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

        assertTensorAlmostEqual(self, xy_grid, xy_grid_expected)


class TestCalcGridIndices(BaseTest):
    def test_calc_grid_indices(self) -> None:
        if torch.__version__ < "1.7.0":
            raise unittest.SkipTest(
                "Skipping grid indices test due to insufficient Torch version."
            )
        xy_grid = torch.arange(0, 2 * 3 * 3).view(3 * 3, 2).float()
        xy_grid = atlas.normalize_grid(xy_grid)
        indices = atlas.calc_grid_indices(xy_grid, grid_size=(2, 2))

        expected_indices = [
            [torch.tensor([0, 1, 2, 3, 4]), torch.tensor([4])],
            [torch.tensor([4]), torch.tensor([4, 5, 6, 7, 8])],
        ]

        for list1, list2 in zip(indices, expected_indices):
            for t1, t2 in zip(list1, list2):
                assertTensorAlmostEqual(self, t1, t2)

    def test_calc_grid_indices_extent(self) -> None:
        if torch.__version__ < "1.7.0":
            raise unittest.SkipTest(
                "Skipping grid indices extent test due to insufficient Torch version."
            )
        xy_grid = torch.arange(0, 2 * 5 * 5).view(5 * 5, 2).float()
        xy_grid = atlas.normalize_grid(xy_grid)
        indices = atlas.calc_grid_indices(
            xy_grid, grid_size=(1, 1), x_extent=(1.0, 2.0), y_extent=(1.0, 2.0)
        )
        assertTensorAlmostEqual(self, indices[0][0], torch.tensor([24]), 0)


class TestExtractGridVectors(BaseTest):
    def test_extract_grid_vectors(self) -> None:
        if torch.__version__ < "1.7.0":
            raise unittest.SkipTest(
                "Skipping extract grid vectors test due to insufficient Torch version."
            )
        grid_size = (2, 2)
        raw_activ = torch.arange(0, 4 * 3 * 3).view(3 * 3, 4).float()
        xy_grid = torch.arange(0, 2 * 3 * 3).view(3 * 3, 2).float()
        xy_grid = atlas.normalize_grid(xy_grid)
        grid_indices = atlas.calc_grid_indices(xy_grid, grid_size=grid_size)

        vecs, vec_coords = atlas.extract_grid_vectors(
            grid_indices, raw_activ, grid_size=grid_size, min_density=2
        )

        expected_vecs = torch.tensor([[8.0, 9.0, 10.0, 11.0], [24.0, 25.0, 26.0, 27.0]])
        expected_coords = [(0, 0, 5), (1, 1, 5)]

        assertTensorAlmostEqual(self, vecs, expected_vecs)
        self.assertEqual(vec_coords, expected_coords)


class TestCreateAtlasVectors(BaseTest):
    def test_create_atlas_vectors(self) -> None:
        if torch.__version__ < "1.7.0":
            raise unittest.SkipTest(
                "Skipping create atlas vectors test due to insufficient Torch version."
            )
        raw_activ = torch.arange(0, 4 * 3 * 3).view(3 * 3, 4).float()
        xy_grid = torch.arange(0, 2 * 3 * 3).view(3 * 3, 2).float()
        vecs, vec_coords = atlas.create_atlas_vectors(
            xy_grid, raw_activ, grid_size=(2, 2), min_density=2, normalize=True
        )

        expected_vecs = torch.tensor([[8.0, 9.0, 10.0, 11.0], [24.0, 25.0, 26.0, 27.0]])
        expected_coords = [(0, 0, 5), (1, 1, 5)]

        assertTensorAlmostEqual(self, vecs, expected_vecs)
        self.assertEqual(vec_coords, expected_coords)

    def test_create_atlas_vectors_diff_grid_sizes(self) -> None:
        if torch.__version__ < "1.7.0":
            raise unittest.SkipTest(
                "Skipping create atlas vectors test due to insufficient Torch version."
            )
        grid_size = (2, 3)
        raw_activ = torch.arange(0, 4 * 5 * 4).view(5 * 4, 4).float()
        xy_grid = torch.arange(0, 2 * 5 * 4).view(5 * 4, 2).float()

        vecs, vec_coords = atlas.create_atlas_vectors(
            xy_grid, raw_activ, grid_size=grid_size, min_density=4, normalize=True
        )

        expected_vecs = torch.tensor(
            [[12.0, 13.0, 14.0, 15.0], [64.0, 65.0, 66.0, 67.0]]
        )
        expected_coords = [(0, 0, 7), (1, 2, 7)]

        assertTensorAlmostEqual(self, vecs, expected_vecs)
        self.assertEqual(vec_coords, expected_coords)


class TestCreateAtlas(BaseTest):
    def test_create_atlas(self) -> None:
        img_list = [torch.ones(1, 3, 4, 4)] * 2
        expected_coords = [(0, 0), (1, 1)]
        canvas = atlas.create_atlas(img_list, expected_coords, grid_size=(2, 2))
        assertTensorAlmostEqual(self, canvas, torch.ones_like(canvas))


if __name__ == "__main__":
    unittest.main()
