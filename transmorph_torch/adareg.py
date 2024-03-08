import torch

def compute_reg_mask(y_true, y_pred, mask_bgd, c=50):
    """https://arxiv.org/pdf/1903.07309.pdf

    Args:
        y_true (_type_): _description_
        y_pred (_type_): _description_
        mask_bgd (_type_): _description_
        a0 (_type_): _description_
        b0 (_type_): _description_

    Returns:
        _type_: _description_
    """

    y_true = torch.mul(y_true, mask_bgd)
    y_pred = torch.mul(y_pred, mask_bgd)
    

    residual = torch.abs(y_true - y_pred) # 4 x 1 x 64 x 64 x 64 or 4 x 1 x 128 x 128

    # compute number of voxels excluding background
    sum_dim = tuple(range(1, len(mask_bgd.shape))) #tuple(1, len(mask_bgd)) gives (1,2,3,4) if 3D and (1,2,3) if 2D
    num_voxels = torch.sum(mask_bgd, sum_dim) # batchsize x 1

    mean = torch.sum(residual, sum_dim) / num_voxels # 4 x 1

    mean_shape = (int(mean.shape[0]), ) + (1, ) * (len(residual.shape) - int(len(mean.shape))) # (4, 1, 1, 1, 1)

    factor = c * residual * mean.view(mean_shape)

    alpha = torch.exp(-factor)

    return alpha