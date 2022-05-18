#!/usr/bin/env python3

import hashlib
import torch

from os import path

from tests.helpers.basic import BaseTest


ASSET_URL = "https://pytorch.s3.amazonaws.com/models/captum/"
TUTORIAL_ASSET_URL = "https://pytorch-tutorial-assets.s3.amazonaws.com/captum/"


def _calc_checksum(filename: str, hash_algorithm: str = "sha512"):
    alg = hashlib.new(hash_algorithm)
    with open(filename, 'rb') as f:
        for chunk in iter(lambda: f.read(128 * alg.block_size), b''):
            alg.update(chunk)
    return alg.hexdigest()


class TestLoadAtlasAssetFiles(BaseTest):
    def test_load_atlas_m4c_activation_samples(self) -> None:
        filename = "inceptionv1_mixed4c_relu_samples_activations.pt"
        samples = torch.hub.load_state_dict_from_url(
            url=path.join(TUTORIAL_ASSET_URL, filename),
            progress=False,
        )
        self.assertEqual(list(samples.shape), [100000, 512])

        file_path = path.join(torch.hub.get_dir(), "checkpoints" , filename)
        checksum = _calc_checksum(file_path)
        expected_checksum = (
            "f8480804891fd098860c76caa34574f4e3eab0bbff9281fa61d4f055789ff40122"
            + "ed6b53754bd531988c4dc207d81b641a9602e5da8349470e567a399316da8f"
        )
        self.assertEqual(checksum, expected_checksum)

    def test_load_atlas_m4c_attribution_samples(self) -> None:
        filename = "inceptionv1_mixed4c_relu_samples_attributions.pt"
        samples = torch.hub.load_state_dict_from_url(
            url=path.join(TUTORIAL_ASSET_URL, filename),
            progress=False,
        )
        self.assertEqual(list(samples.shape), [100000, 1008])

        file_path = path.join(torch.hub.get_dir(), "checkpoints" , filename)        
        checksum = _calc_checksum(file_path)
        expected_checksum = (
            "8c58e9b7a6225ae8635755b6804275063f65281314b72257c56bb9e0381f070179"
            + "d9d9d25d728f529c12f1775eb49035dad706dee8f9b55e925759d3876106cb"
        )
        self.assertEqual(checksum, expected_checksum)

    def test_load_atlas_m5b_activation_samples(self) -> None:
        filename = "inceptionv1_mixed5b_relu_samples_activations.pt"
        samples = torch.hub.load_state_dict_from_url(
            url=path.join(TUTORIAL_ASSET_URL, filename),
            progress=False,
        )
        self.assertEqual(list(samples.shape), [400000, 1024])

        file_path = path.join(torch.hub.get_dir(), "checkpoints" , filename)
        checksum = _calc_checksum(file_path)
        expected_checksum = (
            "32d6c472a9a1366486cb6cc7375c461133035f7595b53ce7ecbcbc5160c19efbb5"
            + "ccbd1c63a7aadf572cad4de62eaf878d699e8c540a38018a074985807d80d7"
        )
        self.assertEqual(checksum, expected_checksum)


    def test_load_atlas_m5b_attribution_samples(self) -> None:
        filename = "inceptionv1_mixed5b_relu_samples_attributions.pt"
        samples = torch.hub.load_state_dict_from_url(
            url=path.join(TUTORIAL_ASSET_URL, filename),
            progress=False,
        )
        self.assertEqual(list(samples.shape), [400000, 1008])

        file_path = path.join(torch.hub.get_dir(), "checkpoints" , filename)
        checksum = _calc_checksum(file_path)
        expected_checksum = (
            "c7abf3485a676897c26e82fc5ac435f7fc508b145adad922d5fca611f6317c8c93"
            + "900e989802a1af7e63bbf339d2bff7779bbb2c31f29fd7021cede4fc23e979"
        )
        self.assertEqual(checksum, expected_checksum)


class TestLoadCLIPAssetFiles(BaseTest):
    def test_load_clip_facet_weights(self) -> None:
        filename = "clip_resnet50x4_facets.pt"
        facet_weights = torch.hub.load_state_dict_from_url(
            url=path.join(ASSET_URL, filename),
            progress=False,
        )
        for facet in facet_weights:
            for weights in facet:
                self.assertEqual(list(weights.shape), [1, 1280, 18, 18])

        file_path = path.join(torch.hub.get_dir(), "checkpoints" , filename)
        checksum = _calc_checksum(file_path)

        expected_checksum = (
            "77165d9cf39b7ad7d364a37e4c92548efa1510a97c1dfd226669925501bedc9872"
            + "1503d36342bab5ce72f0cc733388945f747bc267bd0de41116fb24f8d00caa"
        )
        self.assertEqual(checksum, expected_checksum)
