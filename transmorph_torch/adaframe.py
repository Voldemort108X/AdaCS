import torch

def compute_scoring_mask(y_true, y_pred, mask_bgd, a0, b0):
    """https://arxiv.org/abs/2106.03010

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
    

    residual = torch.abs(y_true - y_pred)

    # compute number of voxels excluding background
    sum_dim = tuple(range(1, len(mask_bgd.shape))) #tuple(1, len(mask_bgd)) gives (1,2,3,4) if 3D and (1,2,3) if 2D
    num_voxels = torch.sum(mask_bgd, sum_dim) # batchsize x 1

    mean = torch.sum(residual, sum_dim) / num_voxels # 4 x 1
    mean_shape = (int(mean.shape[0]), ) + (1, ) * (len(residual.shape) - int(len(mean.shape))) # (4, 1, 1, 1, 1)
    residual_square_sum = torch.sum((residual - mean.view(mean_shape)) ** 2, sum_dim) # 4 x 1


    std = torch.sqrt(residual_square_sum / num_voxels)


    a = a0 / (mean + 1e-7)
    b = b0 * (1 - torch.cos(3.14 *  mean))

    rho = (residual - mean.view(mean_shape)) / (std.view(mean_shape) + 1e-7)



    alpha_prior = 1 - 1/(1 + torch.exp(-(a.view(mean_shape) * rho - b.view(mean_shape))))


    return alpha_prior, rho, a, b