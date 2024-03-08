import torch
import torch.nn.functional as F
import numpy as np
import math



class MSE:
    """
    Mean squared error loss.
    """

    def loss(self, y_true, y_pred, mask_bgd):
        y_true = torch.mul(y_true, mask_bgd)
        y_pred = torch.mul(y_pred, mask_bgd)
        return torch.mean((y_true - y_pred) ** 2)


class Dice:
    """
    N-D dice for segmentation
    """

    def loss(self, y_true, y_pred):
        ndims = len(list(y_pred.size())) - 2
        vol_axes = list(range(2, ndims + 2))
        top = 2 * (y_true * y_pred).sum(dim=vol_axes)
        bottom = torch.clamp((y_true + y_pred).sum(dim=vol_axes), min=1e-5)
        dice = torch.mean(top / bottom)
        return -dice


class Grad:
    """
    N-D gradient loss.
    """

    def __init__(self, penalty='l1', loss_mult=None):
        self.penalty = penalty
        self.loss_mult = loss_mult

    def loss(self, _, y_pred, scoring_mask, mask_bgd): # log_var should not be used for flow regularization
        dy = torch.abs(y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :])
        dx = torch.abs(y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :])
        dz = torch.abs(y_pred[:, :, :, :, 1:] - y_pred[:, :, :, :, :-1])

        if self.penalty == 'l2':
            dy = dy * dy
            dx = dx * dx
            dz = dz * dz

        d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
        grad = d / 3.0

        if self.loss_mult is not None:
            grad *= self.loss_mult # to compensate the 
        return grad

class Grad_2d:
    """
    N-D gradient loss.
    """

    def __init__(self, penalty='l1', loss_mult=None):
        self.penalty = penalty
        self.loss_mult = loss_mult

    def loss(self, _, y_pred, scoring_mask, mask_bgd): # log_var should not be used for flow regularization
        dy = torch.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :])
        dx = torch.abs(y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1])

        if self.penalty == 'l2':
            dy = dy * dy
            dx = dx * dx

        d = torch.mean(dx) + torch.mean(dy)
        grad = d / 2.0

        if self.loss_mult is not None:
            grad *= self.loss_mult # to compensate the 
        return grad




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


class CoVisWeightedMSE:
    """
    Weighted mean squared error loss.
    """

    def __init__(self, loss_mult=None):
        self.loss_mult = loss_mult

    def loss(self, y_true, y_pred, scoring_mask, mask_bgd):
        
        # y_true = y_true.detach()
        # y_pred = y_pred.detach()

        y_true = torch.mul(y_true, mask_bgd)
        y_pred = torch.mul(y_pred, mask_bgd)
        scoring_mask = torch.mul(scoring_mask, mask_bgd)


        return torch.mean(torch.mul((y_true - y_pred) ** 2, scoring_mask))


class ScoringLoss:
    def loss(self,  y_true, y_pred, scoring_mask, mask_bgd):

        ones_mask = torch.ones(scoring_mask.shape).to(scoring_mask.device)

        scoring_mask = torch.mul(scoring_mask, mask_bgd)
        ones_mask = torch.mul(ones_mask, mask_bgd)


        return torch.mean((scoring_mask - ones_mask) ** 2)


class CoVisTVLoss:
    def __init__(self, ndims):
        self.ndims = ndims

    def compute_tv(self, input):
        # input: BxCxXxYxZ
        dx = torch.abs(input[:, :, 1:, :, :] - input[:, :, :-1, :, :])
        dy = torch.abs(input[:, :, :, 1:, :] - input[:, :, :, :-1, :])
        dz = torch.abs(input[:, :, :, :, 1:] - input[:, :, :, :, :-1])

        d = torch.mean(dx ** 2) + torch.mean(dy ** 2) + torch.mean(dz ** 2)

        return d / 3.0
       
    def compute_tv_2d(self, input):
        # input: B x C x X x Y
        dx = torch.abs(input[:, :, 1:, :] - input[:, :, :-1, :])
        dy = torch.abs(input[:, :, :, 1:] - input[:, :, :, :-1])

        d = torch.mean(dx ** 2) + torch.mean(dy ** 2)

        return d / 2.0

    def loss(self,  y_true, y_pred, scoring_mask, mask_bgd):
        if self.ndims == 3:
            return self.compute_tv(scoring_mask)
        elif self.ndims == 2:
            return self.compute_tv_2d(scoring_mask)


##############
# Adareg loss
##############

class AdaRegMSE:
    """
    Mean squared error loss.
    """

    def loss(self, y_true, y_pred, reg_mask, mask_bgd):
        y_true = torch.mul(y_true, mask_bgd)
        y_pred = torch.mul(y_pred, mask_bgd)
        return torch.mean((y_true - y_pred) ** 2)


class AdaRegGrad:
    """
    N-D gradient loss.
    """

    def __init__(self, penalty='l1', loss_mult=None):
        self.penalty = penalty
        self.loss_mult = loss_mult

    def loss(self, _, y_pred, reg_mask, mask_bgd): # log_var should not be used for flow regularization

        y_pred = torch.mul(y_pred, reg_mask)

        dy = torch.abs(y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :])
        dx = torch.abs(y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :])
        dz = torch.abs(y_pred[:, :, :, :, 1:] - y_pred[:, :, :, :, :-1])

        if self.penalty == 'l2':
            dy = dy * dy
            dx = dx * dx
            dz = dz * dz

        d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
        grad = d / 3.0

        if self.loss_mult is not None:
            grad *= self.loss_mult # to compensate the 
        return grad

class AdaRegGrad_2d:
    """
    N-D gradient loss.
    """

    def __init__(self, penalty='l1', loss_mult=None):
        self.penalty = penalty
        self.loss_mult = loss_mult

    def loss(self, _, y_pred, reg_mask, mask_bgd): # log_var should not be used for flow regularization


        y_pred = torch.mul(y_pred, reg_mask)

        dy = torch.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :])
        dx = torch.abs(y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1])

        if self.penalty == 'l2':
            dy = dy * dy
            dx = dx * dx

        d = torch.mean(dx) + torch.mean(dy)
        grad = d / 2.0

        if self.loss_mult is not None:
            grad *= self.loss_mult # to compensate the 
        return grad
