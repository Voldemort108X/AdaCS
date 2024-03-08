import torch
import torch.nn.functional as F
import numpy as np
import math


class CoVisWeightedMSE:
    """
    Weighted mean squared error loss.
    """

    def __init__(self, loss_mult=None):
        self.loss_mult = loss_mult

    def loss(self, y_true, y_pred, scoring_mask):
        
        # y_true = y_true.detach()
        # y_pred = y_pred.detach()

        # y_true = torch.mul(y_true, mask_bgd)
        # y_pred = torch.mul(y_pred, mask_bgd)
        # scoring_mask = torch.mul(scoring_mask, mask_bgd)


        return torch.mean(torch.mul((y_true - y_pred) ** 2, scoring_mask))


class ScoringLoss:
    def loss(self,  y_true, y_pred, scoring_mask):

        ones_mask = torch.ones(scoring_mask.shape).to(scoring_mask.device)

        # scoring_mask = torch.mul(scoring_mask, mask_bgd)
        # ones_mask = torch.mul(ones_mask, mask_bgd)


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

    def loss(self,  y_true, y_pred, scoring_mask):
        if self.ndims == 3:
            return self.compute_tv(scoring_mask)
        elif self.ndims == 2:
            return self.compute_tv_2d(scoring_mask)
