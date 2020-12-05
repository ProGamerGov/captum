import torch


def image_cov(tensor: torch.Tensor) -> torch.Tensor:
    """
    Calculate a tensor's RGB covariance matrix
    """

    tensor = tensor.reshape(-1, tensor.size(1))
    tensor = tensor - tensor.mean(0, keepdim=True)
    return 1 / (tensor.size(0) - 1) * tensor.T @ tensor


def dataset_color_correlation_matrix(
    loader: torch.utils.data.DataLoader, normalize: bool = False
) -> torch.Tensor:
    """
    Calculate the color correlation matrix for a dataset.
    The color correlation matrix can then used in color decorrelation
    transforms for models trained on the dataset.
    """

    cov_mtx = 0
    for images, _ in loader:
        for b in range(images.size(0)):
            cov_mtx += image_cov(images[b].permute(1, 2, 0))

    cov_mtx = cov_mtx / len(loader.dataset)

    U, S, V = torch.svd(cov_mtx)
    epsilon = 1e-10
    svd_sqrt = U @ torch.diag(torch.sqrt(S + epsilon))
    if normalize:
        svd_sqrt / torch.max(torch.norm(svd_sqrt, dim=0))
    return svd_sqrt
