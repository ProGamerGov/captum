import torch


def rgb_cov(tensor: torch.Tensor) -> torch.Tensor:
    """
    RGB Cov
    """
    tensor = tensor.reshape(-1, tensor.size(1))
    tensor = tensor - tensor.mean(0, keepdim=True)
    return 1 / (tensor.size(0) - 1) * tensor.T @ tensor


def dataset_color_decorrelation_matrix(loader: torch.utils.data.DataLoader) -> torch.Tensor:
    cov_mtx = 0
    for images, _ in loader:
        for b in range(images.size(0)):
            cov_mtx += rgb_cov(images[b].permute(1, 2, 0))

    cov_mtx = cov_mtx / len(loader.dataset)

    U, S, V = torch.svd(cov_mtx)
    epsilon = 1e-10
    svd_sqrt = U @ torch.diag(torch.sqrt(S + epsilon))
    return svd_sqrt
