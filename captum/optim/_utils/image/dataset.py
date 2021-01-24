import os
from typing import Dict, List, Optional

import os
import torch

try:
    from tqdm.auto import tqdm
except (ImportError, AssertionError):
    print(
        "The tqdm package is required to use captum.optim's"
        + " capture_activation_samples function with progress bar"
    )

from captum.optim._utils.models import collect_activations


def image_cov(tensor: torch.Tensor) -> torch.Tensor:
    """
    Calculate a tensor's RGB covariance matrix
    """

    tensor = tensor.reshape(-1, 3)
    tensor = tensor - tensor.mean(0, keepdim=True)
    return 1 / (tensor.size(0) - 1) * tensor.T @ tensor


def dataset_cov_matrix(loader: torch.utils.data.DataLoader) -> torch.Tensor:
    """
    Calculate the covariance matrix for an image dataset.
    """

    cov_mtx = torch.zeros(3, 3)
    for images, _ in loader:
        assert images.dim() == 4
        for b in range(images.size(0)):
            cov_mtx = cov_mtx + image_cov(images[b].permute(1, 2, 0))
    cov_mtx = cov_mtx / len(loader.dataset)  # type: ignore
    return cov_mtx


def cov_matrix_to_klt(
    cov_mtx: torch.Tensor, normalize: bool = False, epsilon: float = 1e-10
) -> torch.Tensor:
    """
    Convert a cov matrix to a klt matrix.
    """

    U, S, V = torch.svd(cov_mtx)
    svd_sqrt = U @ torch.diag(torch.sqrt(S + epsilon))
    if normalize:
        svd_sqrt / torch.max(torch.norm(svd_sqrt, dim=0))
    return svd_sqrt


def dataset_klt_matrix(
    loader: torch.utils.data.DataLoader, normalize: bool = False
) -> torch.Tensor:
    """
    Calculate the color correlation matrix, also known as
    a Karhunen-Loève transform (KLT) matrix, for a dataset.
    The color correlation matrix can then used in color decorrelation
    transforms for models trained on the dataset.
    """

    cov_mtx = dataset_cov_matrix(loader)
    return cov_matrix_to_klt(cov_mtx, normalize)


def capture_activation_samples(
    loader: torch.utils.data.DataLoader,
    model: nn.Module,
    targets: List[torch.nn.Module],
    target_names: Optional[List[str]] = None,
    sample_dir: str = "samples",
    num_images: Optional[int] = None,
    samples_per_image: int = 1,
    input_device: torch.device = torch.device("cpu"),
    show_progress: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Capture randomly sampled activations for an image dataset from one or multiple
    target layers.
    Args:
        loader (torch.utils.data.DataLoader): A torch.utils.data.DataLoader
            instance for an image dataset.
        model (nn.Module): A PyTorch model instance.
        targets (list of nn.Module): A list of layers to callect activation samples
            from.
        target_names (list of str, optional): A list of names to use when saving sample
            tensors as files.
        sample_dir (str): Path to where activation samples should be saved.
        num_images (int, optional): How many images to collect samples from.
            Default is to collect samples for every image in the dataset.
        samples_per_image (int): How many samples to collect per image. Default
            is 1 sample per image.
        input_device (torch.device, optional): The device to use for model
            inputs.
        show_progress (bool, optional): Whether or not to show progress.
    """

    def random_sample(activations: torch.Tensor) -> torch.Tensor:
        """
        Randomly sample H & W dimensions of activations with 4 dimensions.
        """
        assert activations.dim() == 4 or activations.dim() == 2

        rnd_samples = []
        for i in range(samples_per_image):
            for b in range(activations.size(0)):
                if activations.dim() == 4:
                    h, w = activations.shape[2:]
                    y = torch.randint(low=1, high=h - 1, size=[1])
                    x = torch.randint(low=1, high=w - 1, size=[1])
                    activ = activations[b, :, y, x]
                elif activations.dim() == 2:
                    activ = activations[b].unsqueeze(1)
                rnd_samples.append(activ)
        return rnd_samples

    if target_names is None:
        target_names = ["target" + str(i) + "_" for i in range(len(targets))]
    assert len(target_names) == len(targets)
    assert os.path.isdir(sample_dir)

    if show_progress:
        total = (
            len(loader.dataset) if num_images is None else num_images  # type: ignore
        )
        pbar = tqdm(total=total, unit=" images")

    image_count = 0
    with torch.no_grad():
        for inputs, _ in loader:
            inputs = inputs.to(input_device)
            image_count += inputs.size(0)

            target_activ_dict = collect_activations(model, targets, inputs)

            [
                torch.save(
                    random_sample(target_activ_dict[t]),
                    os.path.join(sample_dir, +n + "_" + str(batch_count) + ".pt"),
                )
                for t, n in zip(target_activ_dict, target_names)
            ]
            del target_activ_dict

            if show_progress:
                pbar.update(inputs.size(0))

            if num_images is not None:
                if image_count > num_images:
                    break

    if show_progress:
        pbar.close()
