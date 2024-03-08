from medpy import metric
from scipy.io import loadmat
import os
import numpy as np


import torch

def func_computeSegMetrics3D(pred, gt):

    dice = metric.binary.dc(pred, gt)
    jc = metric.binary.jc(pred, gt)
    hd = metric.binary.hd(pred, gt)
    asd = metric.binary.asd(pred, gt)

    return dice, jc, hd, asd

def func_computeImageMetric3D(pred, gt, mask_bgd):
    numOfVoxels = np.sum(mask_bgd)  

    pred = pred * mask_bgd
    gt = gt * mask_bgd
    
    # compute normalized rmse after nomalizing the image with min max
    pred = (pred - np.min(pred)) / (np.max(pred) - np.min(pred))
    gt = (gt - np.min(gt)) / (np.max(gt) - np.min(gt))
    rmse = np.sqrt(np.sum((pred - gt) ** 2)/numOfVoxels)

    # compute normalized cross correlation
    ncc = np.corrcoef(delete_zeros(pred, mask_bgd), delete_zeros(gt, mask_bgd))[0, 1]
    
    return rmse, ncc


def compute_metric(test_dir):
    dice_list, jc_list, hd_list, asd_list = [], [], [], []

    rmse_list, ncc_list = [], []
    for file_name in os.listdir(test_dir):
        file = loadmat(os.path.join(test_dir, file_name))
        try:
            dice, jc, hd, asd = func_computeSegMetrics3D(file['ES_myo'], file['ES_myo_pred'])
            #rmse, ncc = func_computeImageMetric3D(file['im_ES_pred'], file['im_ES'])

            dice_list.append(dice), jc_list.append(jc), hd_list.append(hd), asd_list.append(asd)
            #rmse_list.append(rmse), ncc_list.append(ncc)
        except:
            continue

    return dice_list, jc_list, hd_list, asd_list#, rmse_list, ncc_list


