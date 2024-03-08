import torch
import torch.nn as nn
import torch.nn.functional as F


class gradientLoss(nn.Module):
    def __init__(self, penalty='l1'):
        super(gradientLoss, self).__init__()
        self.penalty = penalty

    def forward(self, input):
        dH = torch.abs(input[:, :, 1:, :] - input[:, :, :-1, :])
        dW = torch.abs(input[:, :, :, 1:] - input[:, :, :, :-1])
        if(self.penalty == "l2"):
            dH = dH * dH
            dW = dW * dW
        loss = (torch.mean(dH) + torch.mean(dW)) / 2.0
        return loss


class crossCorrelation2D(nn.Module):
    def __init__(self, in_ch, kernel=(9, 9), voxel_weights=None):
        super(crossCorrelation2D, self).__init__()
        self.kernel = kernel
        self.voxel_weight = voxel_weights
        self.filt = (torch.ones([1, in_ch, self.kernel[0], self.kernel[1]])).cuda()

    def forward(self, input, target, scoring_mask, mask_bgd):
        input = torch.mul(input, mask_bgd)
        target = torch.mul(target, mask_bgd)
        # min_max = (-1, 1)
        # target = (target - min_max[0]) / (min_max[1] - min_max[0])  # to range [0,1]

        II = input * input * scoring_mask 
        TT = target * target * scoring_mask 
        IT = input * target * scoring_mask 

        pad = (int((self.kernel[0] - 1) / 2), int((self.kernel[1] - 1) / 2))
        T_sum = F.conv2d(target, self.filt, stride=1, padding=pad)
        I_sum = F.conv2d(input, self.filt, stride=1, padding=pad)
        TT_sum = F.conv2d(TT, self.filt, stride=1, padding=pad)
        II_sum = F.conv2d(II, self.filt, stride=1, padding=pad)
        IT_sum = F.conv2d(IT, self.filt, stride=1, padding=pad)
        kernelSize = self.kernel[0] * self.kernel[1]
        Ihat = I_sum / kernelSize
        That = T_sum / kernelSize

        # cross = (I-Ihat)(J-Jhat)
        cross = IT_sum - Ihat * T_sum - That * I_sum + That * Ihat * kernelSize
        T_var = TT_sum - 2 * That * T_sum + That * That * kernelSize
        I_var = II_sum - 2 * Ihat * I_sum + Ihat * Ihat * kernelSize

        cc = cross * cross / (T_var * I_var + 1e-5)
        loss = -1.0 * torch.mean(cc)
        return loss

class MSE:
    """
    Mean squared error loss.
    """

    def loss(self, y_true, y_pred, mask_bgd):
        y_true = torch.mul(y_true, mask_bgd)
        y_pred = torch.mul(y_pred, mask_bgd)
        return torch.mean((y_true - y_pred) ** 2)
    

class MAE:
    """
    Mean squared error loss.
    """

    def loss(self, y_true, y_pred, mask_bgd):
        y_true = torch.mul(y_true, mask_bgd)
        y_pred = torch.mul(y_pred, mask_bgd)
        return torch.mean(torch.abs(y_true - y_pred))



class WeightedMSE:
    """
    Weighted mean squared error loss.
    """

    def __init__(self, loss_mult=None):
        self.loss_mult = loss_mult

    def loss(self, y_true, y_pred, scoring_mask, mask_bgd):

        scoring_mask = scoring_mask.detach()

        y_true = torch.mul(y_true, mask_bgd)
        y_pred = torch.mul(y_pred, mask_bgd)
        scoring_mask = torch.mul(scoring_mask, mask_bgd)


        return torch.mean(torch.mul((y_true - y_pred) ** 2, scoring_mask))


class AdaRegMSE:
    """
    Mean squared error loss.
    """

    def loss(self, y_true, y_pred, reg_mask, mask_bgd):
        y_true = torch.mul(y_true, mask_bgd)
        y_pred = torch.mul(y_pred, mask_bgd)
        return torch.mean((y_true - y_pred) ** 2)


class AdaReggradientLoss(nn.Module):
    def __init__(self, penalty='l1'):
        super(AdaReggradientLoss, self).__init__()
        self.penalty = penalty

    def forward(self, input, reg_mask):
        input = torch.mul(input, reg_mask)

        dH = torch.abs(input[:, :, 1:, :] - input[:, :, :-1, :])
        dW = torch.abs(input[:, :, :, 1:] - input[:, :, :, :-1])
        if(self.penalty == "l2"):
            dH = dH * dH
            dW = dW * dW
        loss = (torch.mean(dH) + torch.mean(dW)) / 2.0
        return loss
