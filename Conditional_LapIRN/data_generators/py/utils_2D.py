import os
import csv
import functools

# third party imports
import numpy as np
import scipy
from skimage import measure
from scipy.io import loadmat, savemat
import mat73

# local/our imports
import pystrum.pynd.ndutils as nd

# load from original package
from .utils import pad,resize

def im_normalize(im):
    return (im - np.min(im))/(np.max(im) - np.min(im))

def load_imfile_mat(
    filename,
    np_var='vol',
    add_batch_axis=False,
    add_feat_axis=False,
    pad_shape=None,
    resize_factor=1,
    ret_affine=False,
    registration=False
):
    """
    Loads a file in nii, nii.gz, mgz, npz, or npy format. If input file is not a string,
    returns it directly (allows files preloaded in memory to be passed to a generator)

    Parameters:
        filename: Filename to load, or preloaded volume to be returned.
        np_var: If the file is a npz (compressed numpy) with multiple variables,
            the desired variable can be specified with np_var. Default is 'vol'.
        add_batch_axis: Adds an axis to the beginning of the array. Default is False.
        add_feat_axis: Adds an axis to the end of the array. Default is False.
        pad_shape: Zero-pad the array to a target shape. Default is None.
        resize: Volume resize factor. Default is 1
        ret_affine: Additionally returns the affine transform (or None if it doesn't exist).
    """
    if isinstance(filename, str) and not os.path.isfile(filename):
        raise ValueError("'%s' is not a file." % filename)

    if filename.endswith('.mat'):
        try:
            file = loadmat(filename)
        except:
            file = mat73.loadmat(filename)

        # for CAMUS/ACDC dataset
        im_ED = file['im_ED']
        im_ES = file['im_ES']

        if registration:
            myo_ED = file['myo_ED']
            myo_ES = file['myo_ES']


        # normalization
        im_ED = im_normalize(im_ED)
        im_ES = im_normalize(im_ES)

        vol_ED = im_ED
        vol_ES = im_ES
        affine = None

        if registration:
            vol_ED_myo = myo_ED
            vol_ES_myo = myo_ES


    if pad_shape:
        vol_ED, _ = pad(vol_ED, pad_shape)
        vol_ES, _ = pad(vol_ES, pad_shape)

        if registration:
            vol_ED_myo, _ = pad(vol_ED_myo, pad_shape)
            vol_ES_myo, _ = pad(vol_ES_myo, pad_shape)



    if add_feat_axis:
        vol_ED = vol_ED[..., np.newaxis]
        vol_ES = vol_ES[..., np.newaxis]

        if registration:
            vol_ED_myo = vol_ED_myo[..., np.newaxis]
            vol_ES_myo = vol_ES_myo[..., np.newaxis]



    if resize_factor != 1:
        vol_ED = resize(vol_ED, resize_factor)
        vol_ES = resize(vol_ES, resize_factor)

        if registration:
            vol_ED_myo = resize(vol_ED_myo, resize_factor)
            vol_ES_myo = resize(vol_ES_myo, resize_factor)


    if add_batch_axis:
        vol_ED = vol_ED[np.newaxis, ...]
        vol_ES = vol_ES[np.newaxis, ...]

        if registration:
            vol_ED_myo = vol_ED_myo[np.newaxis, ...]
            vol_ES_myo = vol_ES_myo[np.newaxis, ...]

    # print(vol_ED.shape)
    # print(vol_ES.shape)
    if registration:
        return (vol_ED, vol_ES, vol_ED_myo, vol_ES_myo, affine) if ret_affine else (vol_ED, vol_ES, vol_ED_myo, vol_ES_myo)
    else:
        return (vol_ED, vol_ES, affine) if ret_affine else (vol_ED, vol_ES)