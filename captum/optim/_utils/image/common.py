from typing import Optional, Tuple

import torch

def get_neuron_pos(
    H: int, W: int, x: Optional[int] = None, y: Optional[int] = None
) -> Tuple[int, int]:
    if x is None:
        _x = W // 2
    else:
        assert x < W
        _x = x

    if y is None:
        _y = H // 2
    else:
        assert y < H
        _y = y
    return _x, _y


def nchannels_to_rgb(x: torch.Tensor, warp: bool = True) -> torch.Tensor:
    """
    Convert an NCHW image with n channels into a 3 channel RGB image.

    Args:
        x (torch.Tensor):  Image tensor to transform into RGB image.
        warp (bool, optional):  Whether or not to make colors more distinguishable.
            Default: True
    Returns:
        *tensor* RGB image
    """

    def hue_to_rgb(angle: float) -> torch.Tensor:
        """
        Create an RGB unit vector based on a hue of the input angle.
        """

        angle = angle - 360 * (angle // 360)
        colors = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.7071, 0.7071, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.7071, 0.7071],
                [0.0, 0.0, 1.0],
                [0.7071, 0.0, 0.7071],
            ]
        )

        idx = math.floor(angle / 60)
        d = (angle - idx * 60) / 60

        if warp:

            def adj(x: float) -> float:
                return math.sin(x * math.pi / 2)

            d = adj(d) if idx % 2 == 0 else 1 - adj(1 - d)

        vec = (1 - d) * colors[idx] + d * colors[(idx + 1) % 6]
        return vec / torch.norm(vec)

    assert x.dim() == 4

    if (x < 0).any():
        x = posneg(x.permute(0, 2, 3, 1), -1).permute(0, 3, 1, 2)

    rgb = torch.zeros(1, 3, x.size(2), x.size(3), device=x.device)
    nc = x.size(1)
    for i in range(nc):
        rgb = rgb + x[:, i][:, None, :, :]
        rgb = rgb * hue_to_rgb(360 * i / nc).to(device=x.device)[None, :, None, None]

    rgb = rgb + torch.ones(x.size(2), x.size(3))[None, None, :, :] * (
        torch.sum(x, 1)[:, None] - torch.max(x, 1)[0][:, None]
    )
    return (rgb / (1e-4 + torch.norm(rgb, dim=1, keepdim=True))) * torch.norm(
        x, dim=1, keepdim=True
    )
