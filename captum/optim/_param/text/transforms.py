from typing import Optional, List, Union
import torch


try:
    from torchtext.transforms import CLIPTokenizer as CLIPTokenizer_TorchText
except ImportError:
    print("torchtext >=0.12.0 is required to use Captum's Optim ClipTokenizer")


class CLIPTokenizer(torch.nn.Module):
    """
    This module allows individuals to use torchtext's CLIP tokenizer with a wrapper
    that handles context_length padding, special start and end tokens, and to tensor
    conversions. This module also supports JIT.

    See here for more details:
    https://pytorch.org/text/main/transforms.html#torchtext.transforms.CLIPTokenizer

    The torchtext CLIPTokenizer is based on these implementations:
    https://github.com/openai/CLIP/blob/main/clip/simple_tokenizer.py
    https://github.com/mlfoundations/open_clip/blob/main/src/clip/tokenizer.py
    """

    __constants__ = [
        "context_length",
        "start_token",
        "end_token",
        "_merges_path",
        "_num_merges",
        "padding_value",
    ]

    def __init__(
        self,
        merges_path: Optional[str] = None,
        context_length: int = 77,
        start_token: Optional[str] = "<|startoftext|>",
        end_token: Optional[str] = "<|endoftext|>",
        pretrained_merges: bool = True,
        num_merges: Optional[int] = None,
        padding_value: int = 0,
    ) -> None:
        """
        Args:

            merges_path (str, optional): Path to file containing the merges, or where
                to save the merges file if pretrained_merges is set to True. The
                torch.hub.get_dir() function will be used to get the directory if set
                to None, resulting in a path of: <PATH_TO_HUB_DIR>/vocab.
                Default: None
            context_length (int, optional): The required context length for the model.
                Inputs with lengths less than context_length will be padded with
                zeros.
                Default: 77
            start_token (str, optional): The starting token to place in front of each
                text input. Set to None for no start token.
                Default: "<|startoftext|>"
            end_token (str, optional): The ending token to place at the end of each
                text input. Set to None for no end token.
                Default: "<|endoftext|>"
            pretrained_merges (bool, optional): Whether or not to download merges for
                the pretrained CLIP model.
                Default: True
            num_merges (int, optional): The number of lines to use from the merges
                file. Set to None for all lines.
                Default: None
            padding_value (int, optional): An integer value to use for padding token
                sets to the desired context_length.
                Default: 0
        """
        super().__init__()
        self.context_length = context_length
        self.start_token = start_token + " " if start_token is not None else ""
        self.end_token = " " + end_token if end_token is not None else ""

        if pretrained_merges:
            merges_path = self._download_clip_bpe_merges(file_dir=merges_path)
        else:
            assert merges_path is not None

        self._num_merges = num_merges
        self._merges_path = merges_path
        self.clip_tokenizer_module = CLIPTokenizer_TorchText(
            merges_path=merges_path, num_merges=num_merges
        )
        self.padding_value = padding_value

    @staticmethod
    @torch.jit.ignore
    def _download_clip_bpe_merges(file_dir: Optional[str] = None) -> str:
        """
        Download a copy of CLIP's BPE merges for the first 48895 lines of the
        'bpe_simple_vocab_16e6.txt.gz' file from: https://github.com/openai/CLIP.

        The BPE merges file will not be downloaded if it already exists in the
        specified directory.

        Args:

            file_dir (str, optional): Optionally provide a location to save the
                file to. The torch.hub.get_dir() function will be used to get the
                directory if set to None, resulting in a path
                of: <PATH_TO_HUB_DIR>/vocab.
                Default: None

            Returns:
                filename (str): The path to the downloaded file with the filename.
        """
        from os import path, makedirs
        import requests

        url = (
            "https://pytorch.s3.amazonaws.com/models/captum/"
            + "clip_bpe_simple_vocab_48895.txt"
        )
        if file_dir is None:
            file_dir = path.join(torch.hub.get_dir(), "vocab")
        else:
            assert not path.isfile(file_dir)

        filename = path.join(file_dir, path.basename(url))

        # Create dir if it doesn't exist
        if file_dir != "" and not path.isdir(file_dir):
            makedirs(file_dir)

        if not path.isfile(filename):
            print("Downloading: '{}' to '{}'\n".format(path.basename(url), file_dir))
            file = requests.get(url)
            with open(filename, "wb") as f:
                f.write(file.content)
        return filename

    @torch.jit.ignore
    def decode(
        self, x: torch.Tensor, include_special_tokens: bool = False
    ) -> List[List[str]]:
        """
        Decode token values into their corresponding string values.

        Based on the implementations used by OpenAI & TorchText:
        https://github.com/openai/gpt-2/blob/master/src/encoder.py
        https://github.com/pytorch/text/blob/main/torchtext/transforms.py

        Args:

            x (torch.Tensor): A set of tokens stacked across the batch dimension.
            include_special_tokens (bool, optional): Whether or not to included added
                special tokens in the output.
                Default: False

        Returns:
            token_str (list of list of str): A set of strings that correspond to the
                token values in the input tensor.
        """
        assert x.dim() == 2
        with open(self._merges_path, "r", encoding="utf-8") as f:
            bpe_merges = f.read().split("\n")[1:]
        num_merges = self._num_merges or len(bpe_merges)

        # Setup vocab Unicode values
        # Unicode values from "!" to "~", "¡" to "¬", "®" to "ÿ"
        # Lowercase & uppercase are treated as the same character
        bpe_v = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
        bpe_keys = bpe_v + list(range(0, 33)) + list(range(127, 161)) + [173]
        bpe_vocab = [chr(c) for c in bpe_v + [256 + n for n in list(range(0, 68))]]
        byte_decoder = dict(zip(bpe_vocab, bpe_keys))

        bpe_vocab += [v + "</w>" for v in bpe_vocab]
        # Add vocab merges from file
        bpe_vocab += [
            "".join(merge_pair.split()) for merge_pair in bpe_merges[:num_merges]
        ]

        # Handle special tokens
        if self.start_token != "":
            bpe_vocab += [self.start_token.strip()]
        if self.end_token != "":
            bpe_vocab += [self.end_token.strip()]

        decoder = dict(zip(range(len(bpe_vocab)), bpe_vocab))

        # Decode tokens
        x = [[t.tolist() for t in b] for b in x]
        x = [[i for i in b if i != self.padding_value] for b in x]
        token_str = ["".join([decoder[t] for t in ts]) for ts in x]
        token_str = [bytearray([byte_decoder[t] for t in ts]) for ts in token_str]
        token_str = [
            ts.decode("utf-8", errors="replace").replace("</w>", " ").strip()
            for ts in token_str
        ]
        if self.start_token != "" and not include_special_tokens:
            token_str = [s.replace(self.start_token.strip(), "") for s in token_str]
        if self.end_token != "" and not include_special_tokens:
            token_str = [s.replace(self.end_token.strip(), "") for s in token_str]
        return [s.strip() for s in token_str]

    def forward(self, x: Union[str, List[str]]) -> torch.Tensor:
        """
        Args:

            x (str or list of str): Text values to be converted to tokenized tensors.

        Returns:
            tokens (torch.Tensor): A tensor containing each set of tokens stacked
                across the batch dimension.
        """
        x = [x] if isinstance(x, str) else x

        # Optionally add start & end tokens to inputs
        x = [self.start_token + s + self.end_token for s in x]

        # Tokenize the text strings
        tokens = self.clip_tokenizer_module(x)

        # Refine 'tokens' Type from Any to List[List[str]] in JIT
        assert torch.jit.isinstance(tokens, List[List[str]])
        assert all([len(t) <= self.context_length for t in tokens])

        # Convert str tokens to tensor values & apply zeros padding
        p = self.padding_value
        tokens = [
            [int(t) for t in token_set] + ([p] * (self.context_length - len(token_set)))
            for token_set in tokens
        ]
        return torch.as_tensor(tokens).int()


__all__ = ["CLIPTokenizer"]
