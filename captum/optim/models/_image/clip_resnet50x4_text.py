from typing import Optional

import math
import torch
from torch import nn


GS_SAVED_WEIGHTS_URL = "clip_rn50x4_text.pt"


def clip_resnet50x4_text(
    pretrained: bool = False,
    progress: bool = True,
    model_path: Optional[str] = None,
    **kwargs
) -> "CLIP_ResNet50x4Text":
    """
    The text portion of OpenAI's ResNet 50x4 CLIP model from 'Learning Transferable
    Visual Models From Natural Language Supervision': https://arxiv.org/abs/2103.00020

    See here for more details: https://github.com/openai/CLIP

    Args:

        pretrained (bool, optional): If True, returns a pre-trained model.
            Default: False
        progress (bool, optional): If True, displays a progress bar of the download to
            stderr
            Default: True
        model_path (str, optional): Optional path for the model file.
            Default: None

    Returns:
        **CLIP_ResNet50x4Text** (CLIP_ResNet50x4Text): A CLIP ResNet 50x4 model's text
            portion.
    """
    if pretrained:
        model = CLIP_ResNet50x4Text(**kwargs)

        if model_path is None:
            state_dict = torch.hub.load_state_dict_from_url(
                GS_SAVED_WEIGHTS_URL, progress=progress, check_hash=False
            )
        else:
            state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict)
        return model

    return CLIP_ResNet50x4Text(**kwargs)


class CLIP_ResNet50x4Text(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        attn_mask = torch.empty(77, 77).fill_(float("-inf")).triu_(1)
        self.transformer = nn.Sequential(
            *[ResidualAttentionBlock(640, 10, attn_mask) for _ in range(12)]
        )
        self.token_embedding = nn.Embedding(49408, 640)
        self.positional_embedding = nn.Parameter(torch.empty(77, 640))
        self.ln_final = nn.LayerNorm(640)

        self.text_projection = nn.Parameter(torch.empty(640, 640))
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))

    def forward(self, text: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(text)
        x = x + self.positional_embedding.to(device=x.device, dtype=x.dtype)
        x = self.transformer(x.permute(1, 0, 2)).permute(1, 0, 2)
        x = self.ln_final(x)
        return x[
            torch.arange(x.shape[0]), text.argmax(dim=-1)
        ] @ self.text_projection.to(device=x.device, dtype=x.dtype)


class QuickGELU(nn.Module):
    """
    OpenAI's model use a slightly different GELU than PyTorch's default GELU.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, width: int, num_heads: int, attn_mask: torch.Tensor) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(width, num_heads)
        self.ln_1 = nn.LayerNorm(width)
        self.mlp = nn.Sequential(
            nn.Linear(width, width * 4), QuickGELU(), nn.Linear(width * 4, width)
        )
        self.ln_2 = nn.LayerNorm(width)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor) -> torch.Tensor:
        attn_mask = self.attn_mask.to(device=x.device, dtype=x.dtype)
        return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask)[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))
