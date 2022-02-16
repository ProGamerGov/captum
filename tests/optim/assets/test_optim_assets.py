#!/usr/bin/env python3
import torch

from tests.helpers.basic import BaseTest


ASSET_URL = "https://pytorch-tutorial-assets.s3.amazonaws.com/captum/"


class TestOptimLoadAssetFiles(BaseTest):
    def test_load_atlas_m4c_activation_samples(self) -> None:
        samples = torch.hub.load_state_dict_from_url(
            url=ASSET_URL + "inceptionv1_mixed4c_relu_samples_activations.pt",
            progress=False,
        )
        self.assertEqual(list(samples.shape), [100000, 512])

    def test_load_atlas_m4c_attribution_samples(self) -> None:
        samples = torch.hub.load_state_dict_from_url(
            url=ASSET_URL + "inceptionv1_mixed4c_relu_samples_attributions.pt",
            progress=False,
        )
        self.assertEqual(list(samples.shape), [100000, 1008])

    def test_load_atlas_m5b_activation_samples(self) -> None:
        samples = torch.hub.load_state_dict_from_url(
            url=ASSET_URL + "inceptionv1_mixed5b_relu_samples_activations.pt",
            progress=False,
        )
        self.assertEqual(list(samples.shape), [400000, 1024])

    def test_load_atlas_m5b_attribution_samples(self) -> None:
        samples = torch.hub.load_state_dict_from_url(
            url=ASSET_URL + "inceptionv1_mixed5b_relu_samples_attributions.pt",
            progress=False,
        )
        self.assertEqual(list(samples.shape), [400000, 1008])
