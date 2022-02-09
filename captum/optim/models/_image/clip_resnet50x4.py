from typing import Optional, Type
from warnings import warn

import torch
from torch import nn

from captum.optim.models._common import RedirectedReluLayer, SkipLayer

GS_SAVED_WEIGHTS_URL = "clip_rn50x4_visual.pt"


def clip_resnet50x4_visual(
    pretrained: bool = False,
    progress: bool = True,
    model_path: Optional[str] = None,
    **kwargs
) -> "CLIP_ResNet50x4":
    """
    The visual portion of OpenAI's ResNet 50x4 CLIP model from 'Learning Transferable
    Visual Models From Natural Language Supervision': https://arxiv.org/abs/2103.00020

    AvgPool2d layers were replaced with AdaptiveAvgPool2d to allow for any input size.

    https://github.com/openai/CLIP

    Args:

        pretrained (bool, optional): If True, returns a pre-trained model.
            Default: False
        progress (bool, optional): If True, displays a progress bar of the download to
            stderr
            Default: True
        model_path (str, optional): Optional path for the model file.
            Default: None
        replace_relus_with_redirectedrelu (bool, optional): If True, return pretrained
            model with Redirected ReLU in place of ReLU layers.
            Default: *True* when pretrained is True otherwise *False*
        use_linear_modules_only (bool, optional): If True, return model
            with all nonlinear layers replaced with linear equivalents.
            Default: False
        transform_input (bool, optional): If True, preprocesses the input according to
            the method with which it was trained.
            Default: False

    Returns:
        **CLIP_ResNet50x4** (CLIP_ResNet50x4): An CLIP ResNet 50x4 model's visual
            portion.
    """
    if pretrained:
        if "transform_input" not in kwargs:
            kwargs["transform_input"] = True
        if "replace_relus_with_redirectedrelu" not in kwargs:
            kwargs["replace_relus_with_redirectedrelu"] = True
        if "use_linear_modules_only" not in kwargs:
            kwargs["use_linear_modules_only"] = False

        model = CLIP_ResNet50x4(**kwargs)

        if model_path is None:
            state_dict = torch.hub.load_state_dict_from_url(
                GS_SAVED_WEIGHTS_URL, progress=progress, check_hash=False
            )
        else:
            state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict)
        return model

    return CLIP_ResNet50x4(**kwargs)


class CLIP_ResNet50x4(nn.Module):
    __constants__ = ["transform_input"]

    def __init__(
        self,
        transform_input: bool = False,
        replace_relus_with_redirectedrelu: bool = False,
        use_linear_modules_only: bool = False,
    ) -> None:
        """
        Args:

            replace_relus_with_redirectedrelu (bool, optional): If True, return
                pretrained model with Redirected ReLU in place of ReLU layers.
                Default: False
            use_linear_modules_only (bool, optional): If True, return pretrained
                model with all nonlinear layers replaced with linear equivalents.
                Default: False
            transform_input (bool, optional): If True, preprocesses the input according
                to the method with which it was trained on.
                Default: False
        """
        super().__init__()
        if use_linear_modules_only:
            activ = SkipLayer
        else:
            if replace_relus_with_redirectedrelu:
                activ = RedirectedReluLayer
            else:
                activ = nn.ReLU

        self.transform_input = transform_input
        width = 80

        # Stem layers
        self.conv1 = nn.Conv2d(
            3, width // 2, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.relu1 = activ()
        self.conv2 = nn.Conv2d(
            width // 2, width // 2, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.relu2 = activ()
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.relu3 = activ()
        self.avgpool = nn.AdaptiveAvgPool2d(72)

        # Residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        print("Layer 1")
        self.layer1 = self._make_layer(width, 4, stride=1, pooling=72, activ=activ)
        print("Layer 2")
        self.layer2 = self._make_layer(width * 2, 6, stride=2, pooling=36, activ=activ)
        print("Layer 3")
        self.layer3 = self._make_layer(width * 4, 10, stride=2, pooling=18, activ=activ)
        print("Layer 4")
        self.layer4 = self._make_layer(width * 8, 6, stride=2, pooling=9, activ=activ)

        self.attnpool = AttentionPool2d(
            9, width * 32, num_heads=width * 32 // 64, output_dim=640
        )

    def _make_layer(
        self,
        planes: int = 80,
        blocks: int = 4,
        stride: int = 1,
        pooling: int = 72,
        activ: Type[nn.Module] = nn.ReLU,
    ) -> nn.Module:
        """
        Residual layer creation helper function, based on the heloper function used
        here: https://github.com/openai/CLIP/blob/main/clip/model.py
        """
        print("first layer", self._inplanes)
        layers = [
            Bottleneck(self._inplanes, planes, stride, pooling=pooling, activ=activ)
        ]
        self._inplanes = planes * 4
        print("subsequent layers", self._inplanes)
        for _ in range(1, blocks):
            layers += [Bottleneck(self._inplanes, planes, pooling=pooling, activ=activ)]
        return nn.Sequential(*layers)

    def _transform_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:

            x (torch.Tensor): An input tensor to normalize and scale the values of.

        Returns:
            x (torch.Tensor): A transformed tensor.
        """
        if self.transform_input:
            assert x.dim() == 3 or x.dim() == 4
            if x.min() < 0.0 or x.max() > 1.0:
                warn("Model input has values outside of the range [0, 1].")
            x = x.unsqueeze(0) if x.dim() == 3 else x
            x = x - torch.tensor(
                [0.48145466, 0.4578275, 0.40821073], device=x.device
            ).view(3, 1, 1)
            x = x / torch.tensor(
                [0.26862954, 0.26130258, 0.27577711], device=x.device
            ).view(3, 1, 1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._transform_input(x)

        # Stem layers
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.avgpool(x)

        # Residual layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.attnpool(x)
        return x


class Bottleneck(nn.Module):
    def __init__(
        self,
        inplanes: int = 80,
        planes: int = 80,
        stride: int = 1,
        pooling: int = 72,
        activ: Type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = activ()

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = activ()

        self.avgpool = nn.AdaptiveAvgPool2d(pooling)

        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu3 = activ()

        self.downsample = None
        if stride > 1 or inplanes != planes * 4:
            self.downsample = nn.Sequential(
                nn.AdaptiveAvgPool2d(pooling),
                nn.Conv2d(inplanes, planes * 4, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * 4),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.downsample is not None:
            identity = self.downsample(x)
        else:
            identity = x.clone()

        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.avgpool(x)
        x = self.bn3(self.conv3(x))

        x = x + identity
        x = self.relu3(x)
        return x


class AttentionPool2d(nn.Module):
    def __init__(
        self,
        spacial_dim: int = 9,
        embed_dim: int = 2560,
        num_heads: int = 40,
        output_dim: int = 640,
    ) -> None:
        super().__init__()
        self.positional_embedding = nn.Parameter(
            torch.randn(spacial_dim**2 + 1, embed_dim) / embed_dim**0.5
        )
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim)
        self.num_heads = num_heads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(*x.shape[:2], -1).permute(2, 0, 1)
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)
        x = x + self.positional_embedding[:, None, :]
        return torch.nn.functional.multi_head_attention_forward(
            query=x,
            key=x,
            value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat(
                [self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]
            ),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False,
        )[0][0]
