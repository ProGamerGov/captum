#!/usr/bin/env python3
import unittest

import torch

import captum.optim._param.text.transforms as transforms
from tests.helpers.basic import BaseTest


try:
    from torchtext.transforms import CLIPTokenizer as CLIPTokenizer_TorchText
    _torchtext_has_clip_tokenizer = True
except ImportError:
    _torchtext_has_clip_tokenizer = False


class TestCLIPTokenizer(BaseTest):
    def test_clip_tokenizer_pretrained_download(self) -> None:
        file_path = path.join(
            torch.hub.get_dir(), "vocab", "clip_bpe_simple_vocab_48895.txt"
        )
        merges_path = transforms.CLIPTokenizer._download_clip_bpe_merges(None)
        self.assertEqual(merges_path, file_path)

    def test_clip_tokenizer_pretrained_download_custom_path(self) -> None:
        file_path = path.join(
            torch.hub.get_dir(), "vocab_test", "clip_bpe_simple_vocab_48895.txt"
        )
        custom_path = path.join(torch.hub.get_dir(), "vocab_test")
        merges_path = transforms.CLIPTokenizer._download_clip_bpe_merges(custom_path)
        self.assertEqual(merges_path, file_path)

    def test_clip_tokenizer_init(self) -> None:
        if not _torchtext_has_clip_tokenizer:
            raise unittest.SkipTest(
                "torchtext >=0.12.0 not found, skipping ClipTokenizer init test"
            )
        clip_tokenizer = transforms.CLIPTokenizer(pretrained_merges=True)

        self.assertEqual(clip_tokenizer.content_length, 77)
        self.assertEqual(clip_tokenizer.start_token, "<|startoftext|> ")
        self.assertEqual(clip_tokenizer.end_token, " <|endoftext|>")

        file_path = path.join(
            torch.hub.get_dir(), "vocab", "clip_bpe_simple_vocab_48895.txt"
        )
        self.assertEqual(clip_tokenizer._merges_path, file_path)

        from torchtext.transforms import CLIPTokenizer as CLIPTokenizer_TorchText

        self.assertIsInstance(
            clip_tokenizer.clip_tokenizer_module, CLIPTokenizer_TorchText
        )

    def test_clip_tokenizer_str_input(self) -> None:
        if not _torchtext_has_clip_tokenizer:
            raise unittest.SkipTest(
                "torchtext >=0.12.0 not found, skipping ClipTokenizer str input test"
            )
        clip_tokenizer = transforms.CLIPTokenizer(pretrained_merges=True)
        text_input_str = "this is a test!"

        text_output = clip_tokenizer(text_input_str)

        content_length = 77
        token_ids = [49406, 589, 533, 320, 1628, 256, 49407]
        padding = [0] * (content_length - len(token_ids))

        token_set = token_ids + padding
        self.assertEqual(text_output.shape, [1, content_length])
        self.assertEqual(text_output[0].tolist(), token_set)

    def test_clip_tokenizer_str_input_content_length_54(self) -> None:
        if not _torchtext_has_clip_tokenizer:
            raise unittest.SkipTest(
                "torchtext >=0.12.0 not found, skipping ClipTokenizer str input"
                + " content_length test"
            )
        clip_tokenizer = transforms.CLIPTokenizer(pretrained_merges=True)
        text_input_str = "this is a test!"

        text_output = clip_tokenizer(text_input_str)

        content_length = 54
        token_ids = [49406, 589, 533, 320, 1628, 256, 49407]
        padding = [0] * (content_length - len(token_ids))

        token_set = token_ids + padding
        self.assertEqual(text_output.shape, [1, content_length])
        self.assertEqual(text_output[0].tolist(), token_set)

    def test_clip_tokenizer_list_str_input(self) -> None:
        if not _torchtext_has_clip_tokenizer:
            raise unittest.SkipTest(
                "torchtext >=0.12.0 not found, skipping ClipTokenizer list str input"
                + " test"
            )
        clip_tokenizer = transforms.CLIPTokenizer(pretrained_merges=True)
        text_input_str = ["this is a test!", "a picture of a cat."]

        text_output = clip_tokenizer(text_input_str)

        content_length = 77
        token_ids = [
            [49406, 589, 533, 320, 1628, 256, 49407],
            [49406, 320, 1674, 539, 320, 2368, 269, 49407],
        ]

        self.assertEqual(text_output.shape, [2, content_length])
        for b, t in enumerate(token_ids):
            padding = [0] * (content_length - len(t))
            token_set = t + padding
            self.assertEqual(text_output[b].tolist(), token_set)

    def test_clip_tokenizer_str_input_decode(self) -> None:
        if not _torchtext_has_clip_tokenizer:
            raise unittest.SkipTest(
                "torchtext >=0.12.0 not found, skipping ClipTokenizer str input"
                + " decode test"
            )
        clip_tokenizer = transforms.CLIPTokenizer(pretrained_merges=True)
        text_input_str = "this is a test!"

        text_output = clip_tokenizer(text_input_str)
        text_output_str = clip_tokenizer.decode(text_output)

        expected_ouput_str = [["this", "is", "a", "test", "!"]]
        self.assertEqual(text_output_str, expected_ouput_str)

    def test_clip_tokenizer_str_input_jit(self) -> None:
        if not _torchtext_has_clip_tokenizer:
            raise unittest.SkipTest(
                "torchtext >=0.12.0 not found, skipping ClipTokenizer str input JIT"
                + " test"
            )
        clip_tokenizer = transforms.CLIPTokenizer(pretrained_merges=True)
        text_input_str = "this is a test!"

        text_output = torch.jit.script(clip_tokenizer(text_input_str))

        content_length = 77
        token_ids = [49406, 589, 533, 320, 1628, 256, 49407]
        padding = [0] * (content_length - len(token_ids))

        token_set = token_ids + padding
        self.assertEqual(text_output.shape, [1, content_length])
        self.assertEqual(text_output[0].tolist(), token_set)
