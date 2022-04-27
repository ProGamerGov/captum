from typing import Optional, List, Union
import torch


try:
    from torchtext.transforms import CLIPTokenizer as CLIPTokenizer_TorchText
except ImportError:
    print("torchtext >=0.12.0 is required to use Captum's Optim ClipTokenizer")


class CLIPTokenizer(torch.nn.Module):
    """
    This module allows individuals to use torchtext's CLIP tokenizer with a wrapper
    that handles content_length padding, special start and end tokens, and to tensor
    conversions. This module also supports JIT.

    See here for more details:
    https://pytorch.org/text/main/transforms.html#torchtext.transforms.CLIPTokenizer

    The torchtext CLIPTokenizer is based on these implementations:
    https://github.com/openai/CLIP/blob/main/clip/simple_tokenizer.py
    https://github.com/mlfoundations/open_clip/blob/main/src/clip/tokenizer.py
    """

    __constants__ = ["content_length", "start_token", "end_token", "_merges_path"]

    def __init__(
        self,
        merges_path: Optional[str] = None,
        content_length: int = 77,
        start_token: Optional[str] = "<|startoftext|>",
        end_token: Optional[str] = "<|endoftext|>",
        pretrained_merges: bool = True,
    ) -> None:
        """
        Args:

            merges_path (str, optional): Path to file containing the merges, or where
                to save the merges file if pretrained_merges is set to True. The
                torch.hub.get_dir() function will be used to get the directory if set
                to None, resulting in a path of: <PATH_TO_HUB_DIR>/vocab.
                Default: None
            content_length (int, optional): The required context length for the model.
                Inputs with lengths less than content_length will be padded with
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
        """
        super().__init__()
        self.content_length = content_length
        self.start_token = start_token + " " if start_token is not None else ""
        self.end_token = " " + end_token if end_token is not None else ""

        if pretrained_merges:
            merges_path = self._download_clip_bpe_merges(file_dir=merges_path)
        else:
            assert merges_path is not None

        self._merges_path = merges_path
        self.clip_tokenizer_module = CLIPTokenizer_TorchText(merges_path=merges_path)

    @torch.jit.ignore
    def _download_clip_bpe_merges(self, file_dir: Optional[str] = None) -> str:
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
    def decode(self, x: torch.Tensor) -> List[List[str]]:
        """
        Decode token values into their corresponding string values.

        Based on the implementations used by OpenAI & TorchText:
        https://github.com/openai/gpt-2/blob/master/src/encoder.py
        https://github.com/pytorch/text/blob/main/torchtext/transforms.py

        Args:
            x (torch.Tensor): A set of tokens stacked across the batch dimension.

        Returns:
            token_str (list of list of str): A set of strings that correspond to the
                token values in the input tensor.
        """
        assert x.dim() == 2
        with open(self._merges_path, "r", encoding="utf-8") as f:
            bpe_merges = f.read().split("\n")[1:]

        # Setup vocab Unicode values
        bpe_v = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
        bpe_vocab = [chr(c) for c in bpe_v + [256 + n for n in list(range(0, 68))]]
        bpe_vocab += [v + "</w>" for v in bpe_vocab]
        # Add vocab merges from file
        bpe_vocab += [
            "".join(merge_pair.split()) for merge_pair in bpe_merges[: len(bpe_merges)]
        ]
        special_tokens = [self.start_token.strip(), self.end_token.strip()]
        bpe_vocab += special_tokens
        token_str = [[bpe_vocab[int(i)] for i in b if int(i) != 0] for b in x]
        token_str = [[s.replace("</w>", "").strip() for s in b] for b in token_str]
        return [[s for s in b if s not in special_tokens] for b in token_str]

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
        assert all([len(t) <= self.content_length for t in tokens])

        # Convert str tokens to tensor values & apply zeros padding
        tokens = [
            [int(t) for t in token_set] + ([0] * (self.content_length - len(token_set)))
            for token_set in tokens
        ]
        return torch.as_tensor(tokens).int()


__all__ = ["ClipTokenizer"]
