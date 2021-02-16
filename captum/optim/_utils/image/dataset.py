import os
from typing import List, Optional, Tuple, Union, cast

import torch

try:
    from tqdm.auto import tqdm
except (ImportError, AssertionError):
    print(
        "The tqdm package is required to use captum.optim's"
        + " image dataset functions with progress bar"
    )

from captum.optim._utils.models import collect_activations


def image_cov(tensor: torch.Tensor) -> torch.Tensor:
    """
    Calculate a tensor's RGB covariance matrix
    """

    tensor = tensor.reshape(-1, 3)
    tensor = tensor - tensor.mean(0, keepdim=True)
    return 1 / (tensor.size(0) - 1) * tensor.T @ tensor


def dataset_cov_matrix(
    loader: torch.utils.data.DataLoader,
    show_progress: bool = False,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Calculate the covariance matrix for an image dataset.
    """

    if show_progress:
        pbar = tqdm(total=len(loader.dataset), unit=" images")  # type: ignore

    cov_mtx = cast(torch.Tensor, 0.0)
    for images, _ in loader:
        assert images.dim() == 4
        images = images.to(device)
        for b in range(images.size(0)):
            cov_mtx = cov_mtx + image_cov(images[b].permute(1, 2, 0))

            if show_progress:
                pbar.update(1)

    if show_progress:
        pbar.close()

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
    loader: torch.utils.data.DataLoader,
    normalize: bool = False,
    show_progress: bool = False,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Calculate the color correlation matrix, also known as
    a Karhunen-Loève transform (KLT) matrix, for a dataset.
    The color correlation matrix can then used in color decorrelation
    transforms for models trained on the dataset.
    """

    cov_mtx = dataset_cov_matrix(loader, show_progress=show_progress, device=device)
    return cov_matrix_to_klt(cov_mtx, normalize)


def find_pos_attr(
    logit_activ: torch.Tensor,
    target_activ: torch.Tensor,
    y: Optional[Union[int, torch.Tensor]] = None,
    x: Optional[Union[int, torch.Tensor]] = None,
) -> torch.Tensor:
    assert x is not None and y is not None or x is None and y is None

    zeros = torch.nn.Parameter(torch.zeros_like(logit_activ))
    target_zeros = torch.zeros_like(target_activ)

    if x is not None and y is not None:
        target_zeros[..., y, x] = target_activ[..., y, x]
    else:
        target_zeros = target_activ

    grad_one = torch.autograd.grad(
        outputs=[logit_activ],
        inputs=[target_activ],
        grad_outputs=[zeros],
        create_graph=True,
    )[0]
    logit_attr = torch.autograd.grad(
        outputs=[grad_one],
        inputs=zeros,
        grad_outputs=[target_zeros],
        create_graph=True,
    )[0]
    return torch.argsort(-logit_attr[0])


def capture_activation_samples(
    loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    targets: List[torch.nn.Module],
    target_names: Optional[List[str]] = None,
    sample_dir: str = "",
    num_images: Optional[int] = None,
    samples_per_image: int = 1,
    input_device: torch.device = torch.device("cpu"),
    collect_attributions: bool = False,
    logit_target: Optional[torch.nn.Module] = None,
    show_progress: bool = False,
):
    """
    Capture randomly sampled activations for an image dataset from one or multiple
    target layers.
    Args:
        loader (torch.utils.data.DataLoader): A torch.utils.data.DataLoader
            instance for an image dataset.
        model (nn.Module): A PyTorch model instance.
        targets (list of nn.Module): A list of layers to collect activation samples
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
        collect_attributions (bool, optional): Whether or not to collect attributions
            for samples.
        logit_target (nn.Module, optional): The final layer in the model that
            determines the classes.
        show_progress (bool, optional): Whether or not to show progress.
    """

    if collect_attributions:
        logit_target == list(model.children())[len(list(model.children())) - 1 :][
            0
        ] if logit_target is None else logit_target
        targets += [cast(torch.nn.Module, logit_target)]

    def random_sample(
        activations: torch.Tensor,
        logit_activ: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], List]:
        """
        Randomly sample H & W dimensions of activations with 4 dimensions.
        """
        assert activations.dim() == 4 or activations.dim() == 2

        if collect_attributions:
            sample_attributions: List = []

        activation_samples = []
        for i in range(samples_per_image):
            for b in range(activations.size(0)):
                if activations.dim() == 4:
                    h, w = activations.shape[2:]
                    y = torch.randint(low=1, high=h - 1, size=[1])
                    x = torch.randint(low=1, high=w - 1, size=[1])
                    activ = activations[b, :, y, x]
                    if collect_attributions:
                        attr = find_pos_attr(activations[b : b + 1], logit_activ, y, x)
                elif activations.dim() == 2:
                    activ = activations[b].unsqueeze(1)
                    if collect_attributions:
                        attr = find_pos_attr(activations[b : b + 1], logit_activ)
                activation_samples.append(activ.detach())
                if collect_attributions:
                    sample_attributions.append(attr.detach())
        return activation_samples, sample_attributions

    if target_names is None:
        target_names = ["target" + str(i) + "_" for i in range(len(targets))]

    assert len(target_names) == len(targets)
    assert os.path.isdir(sample_dir)

    if show_progress:
        total = (
            len(loader.dataset) if num_images is None else num_images  # type: ignore
        )
        pbar = tqdm(total=total, unit=" images")

    image_count, batch_count = 0, 0
    with torch.set_grad_enabled(collect_attributions):
        for inputs, _ in loader:
            inputs = inputs.to(input_device)
            image_count += inputs.size(0)
            batch_count += 1

            if collect_attributions:
                target_activ_dict = collect_activations(model, targets, inputs)
                logit_activ = target_activ_dict[logit_target]
                del target_activ_dict[logit_target]
            else:
                target_activ_dict = collect_activations(model, targets, inputs)
                logit_activ = None

            for t, n in zip(target_activ_dict, target_names):
                sample_tensors = random_sample(target_activ_dict[t], logit_activ)
                torch.save(
                    sample_tensors[0],
                    os.path.join(sample_dir, n + "_" + str(batch_count) + ".pt"),
                )
                if collect_attributions:
                    torch.save(
                        sample_tensors[1],
                        os.path.join(
                            sample_dir,
                            n + "_attributions_" + str(batch_count) + ".pt",
                        ),
                    )

            del target_activ_dict

            if show_progress:
                pbar.update(inputs.size(0))

            if num_images is not None:
                if image_count > num_images:
                    break

    if show_progress:
        pbar.close()
