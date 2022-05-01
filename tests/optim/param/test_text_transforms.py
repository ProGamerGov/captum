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

        self.assertEqual(clip_tokenizer.context_length, 77)
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

        context_length = 77
        token_ids = [49406, 589, 533, 320, 1628, 256, 49407]
        padding = [0] * (context_length - len(token_ids))

        token_set = token_ids + padding
        self.assertEqual(text_output.shape, [1, context_length])
        self.assertEqual(text_output[0].tolist(), token_set)

    def test_clip_tokenizer_str_input_context_length_54(self) -> None:
        if not _torchtext_has_clip_tokenizer:
            raise unittest.SkipTest(
                "torchtext >=0.12.0 not found, skipping ClipTokenizer str input"
                + " context_length test"
            )
        clip_tokenizer = transforms.CLIPTokenizer(pretrained_merges=True)
        text_input_str = "this is a test!"

        text_output = clip_tokenizer(text_input_str)

        context_length = 54
        token_ids = [49406, 589, 533, 320, 1628, 256, 49407]
        padding = [0] * (context_length - len(token_ids))

        token_set = token_ids + padding
        self.assertEqual(text_output.shape, [1, context_length])
        self.assertEqual(text_output[0].tolist(), token_set)

    def test_clip_tokenizer_str_input_context_length_padding(self) -> None:
        if not _torchtext_has_clip_tokenizer:
            raise unittest.SkipTest(
                "torchtext >=0.12.0 not found, skipping ClipTokenizer str input test"
            )
        padding_value = -1
        clip_tokenizer = transforms.CLIPTokenizer(
            pretrained_merges=True, padding_value=padding_value
        )
        text_input_str = "this is a test!"

        text_output = clip_tokenizer(text_input_str)

        context_length = 77
        token_ids = [49406, 589, 533, 320, 1628, 256, 49407]
        padding = [padding_value] * (context_length - len(token_ids))

        token_set = token_ids + padding
        self.assertEqual(text_output.shape, [1, context_length])
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

        context_length = 77
        token_ids = [
            [49406, 589, 533, 320, 1628, 256, 49407],
            [49406, 320, 1674, 539, 320, 2368, 269, 49407],
        ]

        self.assertEqual(text_output.shape, [2, context_length])
        for b, t in enumerate(token_ids):
            padding = [0] * (context_length - len(t))
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

        jit_clip_tokenizer = torch.jit.script(clip_tokenizer)
        text_output = jit_clip_tokenizer(text_input_str)

        context_length = 77
        token_ids = [49406, 589, 533, 320, 1628, 256, 49407]
        padding = [0] * (context_length - len(token_ids))

        token_set = token_ids + padding
        self.assertEqual(text_output.shape, [1, context_length])
        self.assertEqual(text_output[0].tolist(), token_set)

    def test_clip_tokenizer_unicode(self) -> None:
        if not _torchtext_has_clip_tokenizer:
            raise unittest.SkipTest(
                "torchtext >=0.12.0 not found, skipping ClipTokenizer unicode test"
            )

        from torchtext import __version__ as torchtext_version

        clip_tokenizer = transforms.CLIPTokenizer(
            pretrained_merges=True, context_length=376
        )

        bpe_v = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
        bpe_vocab = [chr(c) for c in bpe_v + [256 + n for n in list(range(0, 68))]]
        bpe_vocab_str = " ".join(bpe_vocab)
        txt_output = clip_tokenizer(bpe_vocab_str)

        # fmt: off
        expected_tokens = [
            256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270,
            271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285,
            286, 287, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332,
            333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 314, 315,
            316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330,
            331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345,
            346, 347, 348, 349, 10830, 41359, 1950, 126, 353, 20199, 126, 355, 126,
            356, 126, 357, 5811, 126, 359, 14434, 126, 361, 8436, 43593, 6858, 126,
            365, 41175, 126, 367, 12559, 126, 369, 126, 370, 14844, 126, 372, 126,
            373, 28059, 7599, 126, 376, 33613, 126, 378, 17133, 21259, 22229, 127,
            351, 127, 352, 47276, 127, 354, 127, 355, 127, 356, 37761, 4166, 127, 359,
            40520, 127, 361, 23928, 127, 362, 127, 363, 127, 364, 127, 365, 127, 366,
            27733, 127, 368, 127, 369, 37423, 16165, 45598, 127, 373, 36019, 127, 375,
            47177, 127, 377, 127, 378, 127, 509, 21259, 22229, 127, 351, 127, 352,
            47276, 127, 354, 127, 355, 127, 356, 37761, 4166, 127, 359, 40520, 127,
            361, 23928, 127, 362, 127, 363, 127, 364, 127, 365, 127, 366, 27733, 127,
            368, 127, 369, 37423, 127, 371, 45598, 127, 373, 36019, 127, 375, 47177,
            127, 377, 127, 378, 127, 379, 128, 479, 128, 479, 128, 481, 128, 481, 128,
            483, 128, 483, 31719, 31719, 128, 487, 128, 487, 128, 489, 128, 489, 128,
            491, 128, 491, 128, 493, 128, 493, 128, 495, 128, 495, 128, 497, 128, 497,
            128, 499, 128, 499, 128, 501, 128, 501, 128, 503, 128, 503, 128, 505, 128,
            505, 128, 507, 128, 507, 128, 509, 128, 509, 128, 350, 128, 350, 128, 352,
            128, 352, 128, 354, 128, 354, 128, 356, 128, 356, 128, 358, 128, 358, 128,
            360, 128, 360, 128, 511, 128, 511, 128, 363, 128, 363, 328, 16384, 41901,
            72, 329, 72, 329, 128, 369, 128, 369, 128, 371, 128, 371, 128, 372, 128,
            374, 128, 374, 128, 376, 128, 376, 128, 378, 128, 378, 129, 478, 129, 478, 
            129, 480, 129, 480, 129, 482
            ]
        # fmt: on

        if torchtext_version <= "0.12.0":
            # Correct for TorchText bug
            expected_tokens[338:342] = [128, 367, 128, 367]

        self.assertEqual(txt_output[0].tolist()[1:-1], expected_tokens)

    def test_clip_tokenizer_unicode_decode(self) -> None:
        if not _torchtext_has_clip_tokenizer:
            raise unittest.SkipTest(
                "torchtext >=0.12.0 not found, skipping ClipTokenizer unicode decode"
                + " test"
            )

        from torchtext import __version__ as torchtext_version

        # fmt: off
        input_tokens = [
            256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270,
            271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285,
            286, 287, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332,
            333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 314, 315,
            316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330,
            331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345,
            346, 347, 348, 349, 10830, 41359, 1950, 126, 353, 20199, 126, 355, 126,
            356, 126, 357, 5811, 126, 359, 14434, 126, 361, 8436, 43593, 6858, 126,
            365, 41175, 126, 367, 12559, 126, 369, 126, 370, 14844, 126, 372, 126,
            373, 28059, 7599, 126, 376, 33613, 126, 378, 17133, 21259, 22229, 127,
            351, 127, 352, 47276, 127, 354, 127, 355, 127, 356, 37761, 4166, 127, 359,
            40520, 127, 361, 23928, 127, 362, 127, 363, 127, 364, 127, 365, 127, 366,
            27733, 127, 368, 127, 369, 37423, 16165, 45598, 127, 373, 36019, 127, 375,
            47177, 127, 377, 127, 378, 127, 509, 21259, 22229, 127, 351, 127, 352,
            47276, 127, 354, 127, 355, 127, 356, 37761, 4166, 127, 359, 40520, 127,
            361, 23928, 127, 362, 127, 363, 127, 364, 127, 365, 127, 366, 27733, 127,
            368, 127, 369, 37423, 127, 371, 45598, 127, 373, 36019, 127, 375, 47177,
            127, 377, 127, 378, 127, 379, 128, 479, 128, 479, 128, 481, 128, 481, 128,
            483, 128, 483, 31719, 31719, 128, 487, 128, 487, 128, 489, 128, 489, 128,
            491, 128, 491, 128, 493, 128, 493, 128, 495, 128, 495, 128, 497, 128, 497,
            128, 499, 128, 499, 128, 501, 128, 501, 128, 503, 128, 503, 128, 505, 128,
            505, 128, 507, 128, 507, 128, 509, 128, 509, 128, 350, 128, 350, 128, 352,
            128, 352, 128, 354, 128, 354, 128, 356, 128, 356, 128, 358, 128, 358, 128,
            360, 128, 360, 128, 511, 128, 511, 128, 363, 128, 363, 328, 16384, 41901,
            72, 329, 72, 329, 128, 369, 128, 369, 128, 371, 128, 371, 128, 372, 128,
            374, 128, 374, 128, 376, 128, 376, 128, 378, 128, 378, 129, 478, 129, 478, 
            129, 480, 129, 480, 129, 482
            ]
        # fmt: on

        input_tokens = torch.as_tensor([input_tokens])
        clip_tokenizer = transforms.CLIPTokenizer(pretrained_merges=True)
        txt_output_str = clip_tokenizer.decode(input_tokens)

        expected_str = (
            """!"#$%&'()*+,-./0123456789:;<=>?@abcdefghijklmnopqrstuvwxyz[\]^_`abcd"""
            + """efghijklmnopqrstuvwxyz{|}~¡¢£¤¥¦§¨©ª«¬®¯°±²³´µ¶·¸¹º»¼½¾¿àáâãäåæçèé"""
            + """êëìíîïðñòóôõö×øùúûüýþßàáâãäåæçèéêëìíîïðñòóôõö÷øùúûüýþÿāāăăąąććĉĉċċ"""
            + """ččďďđđēēĕĕėėęęěěĝĝğğġġģģĥĥħħĩĩīīĭĭįįi̇ıijijĵĵķķĸĺĺļļľľŀŀłłń"""
        )
        self.assertEqual(len(txt_output_str), 1)
        self.assertEqual(txt_output_str[0].replace(" ", ""), expected_str)
