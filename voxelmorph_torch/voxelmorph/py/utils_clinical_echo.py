import os
import csv
import functools

# third party imports
import numpy as np
import scipy
from skimage import measure
from scipy.io import loadmat, savemat
import mat73
import random

# local/our imports
import pystrum.pynd.ndutils as nd

# load from original package
from .utils import pad,resize

def im_normalize(im):
    return (im - np.min(im))/(np.max(im) - np.min(im))

def load_volfile_mat(
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

        # for ED/ES franes are labelled
        if 'ED_idx' in file.keys():
            im_ED = file['im4D_resize'][:, :, :, int(file['ED_idx'].squeeze())]
            im_ES = file['im4D_resize'][:, :, :, int(file['ES_idx'].squeeze())]
            
            if registration:
                myo_ED = file['epi_ED_resize'] - file['endo_ED_resize']
                myo_ES = file['epi_ES_resize'] - file['endo_ES_resize']

                epi_ED = file['epi_ED_resize']
                epi_ES = file['epi_ES_resize']

                endo_ED = file['endo_ED_resize']
                endo_ES = file['endo_ES_resize']
        
        else:
            # for clinical echo
            numOfFrames = file['im4D_resize'].shape[-1]

            numOfCycle = numOfFrames // 10 # rough estimate of the frames per cycle
            framesPerCycle = numOfFrames // numOfCycle
            # rand_idx = random.randint(0, numOfCycle-1) # pick a random cycle
            rand_idx = 0 # only choose the first cycle since the index is more accurate than the random index
            
            ED_idx = rand_idx * framesPerCycle

            ES_idx = ED_idx + framesPerCycle // 2

            im_ED = file['im4D_resize'][:, :, :, ED_idx]
            im_ES = file['im4D_resize'][:, :, :, ES_idx]

            if registration:
                myo_ED = file['epi4D_resize'][:, :, :, ED_idx] - file['endo4D_resize'][:, :, :, ED_idx]
                myo_ES = file['epi4D_resize'][:, :, :, ES_idx] - file['endo4D_resize'][:, :, :, ES_idx]

                # epi_ED = file['epi4D_resize'][:, :, :, ED_idx]
                # epi_ES = file['epi4D_resize'][:, :, :, ES_idx]

                # endo_ED = file['endo4D_resize'][:, :, :, ED_idx]
                # endo_ES = file['endo4D_resize'][:, :, :, ES_idx]
        
        # normalization
        im_ED = im_normalize(im_ED)
        im_ES = im_normalize(im_ES)

        vol_ED = im_ED
        vol_ES = im_ES
        affine = None

        if registration:
            vol_ED_myo = myo_ED
            vol_ES_myo = myo_ES

            vol_ED_epi = epi_ED
            vol_ES_epi = epi_ES

            vol_ED_endo = endo_ED
            vol_ES_endo = endo_ES

    # if not os.path.isfile(filename):
    #     if ret_affine:
    #         (vol, affine) = filename
    #     else:
    #         vol = filename
    # elif filename.endswith(('.nii', '.nii.gz', '.mgz')):
    #     import nibabel as nib
    #     img = nib.load(filename)
    #     vol = img.get_data().squeeze()
    #     affine = img.affine
    # elif filename.endswith('.npy'):
    #     vol = np.load(filename)
    #     affine = None
    # elif filename.endswith('.npz'):
    #     npz = np.load(filename)
    #     vol = next(iter(npz.values())) if len(npz.keys()) == 1 else npz[np_var]
    #     affine = None
    # else:
    #     raise ValueError('unknown filetype for %s' % filename)

    if pad_shape:
        vol_ED, _ = pad(vol_ED, pad_shape)
        vol_ES, _ = pad(vol_ES, pad_shape)

        if registration:
            vol_ED_myo, _ = pad(vol_ED_myo, pad_shape)
            vol_ES_myo, _ = pad(vol_ES_myo, pad_shape)

            vol_ED_epi, _ = pad(vol_ED_epi, pad_shape)
            vol_ES_epi, _ = pad(vol_ES_epi, pad_shape)

            vol_ED_endo, _ = pad(vol_ED_endo, pad_shape)
            vol_ES_endo, _ = pad(vol_ES_endo, pad_shape)

    if add_feat_axis:
        vol_ED = vol_ED[..., np.newaxis]
        vol_ES = vol_ES[..., np.newaxis]

        if registration:
            vol_ED_myo = vol_ED_myo[..., np.newaxis]
            vol_ES_myo = vol_ES_myo[..., np.newaxis]

            vol_ED_epi = vol_ED_epi[..., np.newaxis]
            vol_ES_epi = vol_ES_epi[..., np.newaxis]
            
            vol_ED_endo = vol_ED_endo[..., np.newaxis]
            vol_ES_endo = vol_ES_endo[..., np.newaxis]

    if resize_factor != 1:
        vol_ED = resize(vol_ED, resize_factor)
        vol_ES = resize(vol_ES, resize_factor)

        if registration:
            vol_ED_myo = resize(vol_ED_myo, resize_factor)
            vol_ES_myo = resize(vol_ES_myo, resize_factor)

            vol_ED_epi = resize(vol_ED_epi, resize_factor)
            vol_ES_epi = resize(vol_ES_epi, resize_factor)
            
            vol_ED_endo = resize(vol_ED_endo, resize_factor)
            vol_ES_endo = resize(vol_ES_endo, resize_factor)

    if add_batch_axis:
        vol_ED = vol_ED[np.newaxis, ...]
        vol_ES = vol_ES[np.newaxis, ...]

        if registration:
            vol_ED_myo = vol_ED_myo[np.newaxis, ...]
            vol_ES_myo = vol_ES_myo[np.newaxis, ...]

            vol_ED_epi = vol_ED_epi[np.newaxis, ...]
            vol_ES_epi = vol_ES_epi[np.newaxis, ...]

            vol_ED_endo = vol_ED_endo[np.newaxis, ...]
            vol_ES_endo = vol_ES_endo[np.newaxis, ...]

    # print(vol_ED.shape)
    # print(vol_ES.shape)
    # if registration:
    #     return (vol_ED, vol_ES, vol_ED_myo, vol_ES_myo, affine) if ret_affine else (vol_ED, vol_ES, vol_ED_myo, vol_ES_myo)
    # else:
    #     return (vol_ED, vol_ES, affine) if ret_affine else (vol_ED, vol_ES)

    if registration:
        return (vol_ED, vol_ES, vol_ED_myo, vol_ES_myo, vol_ED_epi, vol_ES_epi, vol_ED_endo, vol_ES_endo, affine) if ret_affine else (vol_ED, vol_ES, vol_ED_myo, vol_ES_myo, vol_ED_epi, vol_ES_epi, vol_ED_endo, vol_ES_endo)
    else:
        return (vol_ED, vol_ES, affine) if ret_affine else (vol_ED, vol_ES)