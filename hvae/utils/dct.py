"""DCT-related functions."""
import numpy as np
import torch
from scipy.fft import dctn, idctn


def reconstruct_dct(imgs, k):
    """Reconstruct images from their DCT coefficients.
    Args:
        imgs: Tensor of shape (B x C x H x W) with the input images
        k: The number of DCT coefficients to keep
    """
    mask = get_mask(k, imgs.shape[1:])
    return torch.stack(
        [torch.from_numpy(idctn(dctn(img) * mask).astype(float)) for img in imgs]
    )


def get_mask(k: int, input_shape):
    """Mask out all DCT coefficients except the first k.
    Args:
        k: Number of DCT coefficients to keep
        input_shape: Shape of the input image
    """
    i, j = np.indices(input_shape[1:])
    mask = (i + j) <= k
    return np.expand_dims(mask, 0).astype(int)
