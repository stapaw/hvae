"""DCT-related functions."""
import numpy as np
import torch
from scipy.fft import dctn, idctn


class DCTMaskTransform:
    def __init__(self, k, mask_func, input_shape):
        self.mask = mask_func(k, input_shape)
        self.input_shape = input_shape

    def __call__(self, x):
        return torch.from_numpy(idctn(dctn(x.numpy()) * self.mask).astype(np.float32))



def reconstruct_dct(imgs, k):
    """Reconstruct images from their DCT coefficients.
    Args:
        imgs: Tensor of shape (B x C x H x W) with the input images
        k: The number of DCT coefficients to keep
    """
    imgs = imgs.detach().cpu().numpy()
    mask = get_mask(k, imgs.shape[1:])
    return torch.stack(
        [torch.from_numpy(idctn(dctn(img) * mask).astype(np.float32)) for img in imgs]
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
