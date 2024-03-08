import os
import argparse
import glob

# third party
import numpy as np
import nibabel as nib
import torch
from scipy.io import loadmat, savemat
import nrrd
from helper import *

# import voxelmorph with pytorch backend
os.environ['NEURITE_BACKEND'] = 'pytorch'
os.environ['VXM_BACKEND'] = 'pytorch'
import voxelmorph as vxm   # nopep8
from voxelmorph.torch.layers import SpatialTransformer


# parse commandline args
parser = argparse.ArgumentParser()
# parser.add_argument('--moving', required=True, help='moving image (source) filename')
# parser.add_argument('--fixed', required=True, help='fixed image (target) filename')
# parser.add_argument('--moved', required=True, help='warped image output filename')
parser.add_argument('--dataset', required=True, help='Name of the dataset')
parser.add_argument('--test-dir', required=True, help='path to all test files')
parser.add_argument('--result-dir', required=True, help='where to save the result')
parser.add_argument('--model-motion', required=True, help='pytorch model for nonlinear registration')
parser.add_argument('--model-covis', required=True, help='pytorch model for segmentation')
parser.add_argument('--inshape', required=True, nargs='+', type=int, help='input shape of the network e.g. (64, 64, 64) or (128, 128)')
# parser.add_argument('--warp', help='output warp deformation filename')
# parser.add_argument('-g', '--gpu', help='GPU number(s) - if not supplied, CPU is used')
parser.add_argument('--multichannel', action='store_true',
                    help='specify that data has multiple channels')
args = parser.parse_args()

assert args.dataset in ['Echo']
inshape = tuple(args.inshape)
ndims = len(inshape)

if not os.path.exists(args.result_dir):
    os.makedirs(args.result_dir)

# # device handling
# if args.gpu and (args.gpu != '-1'):
#     device = 'cuda'
#     os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
# else:
#     device = 'cpu'
#     os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

device = 'cuda'

test_files = glob.glob(os.path.join(args.test_dir, '*.mat'))
assert len(test_files) > 0

# load moving and fixed images
add_feat_axis = not args.multichannel
# moving = vxm.py.utils.load_volfile(args.moving, add_batch_axis=True, add_feat_axis=add_feat_axis)
# fixed, fixed_affine = vxm.py.utils.load_volfile(
#     args.fixed, add_batch_axis=True, add_feat_axis=add_feat_axis, ret_affine=True)

# load and set up model
# if args.baseline:
#     model = vxm.networks.VxmDenseBaseline.load(args.model, device)
# else:

# stats_dict = torch.load(args.model)
# state_dict = stats_dict['config']['inshape'] = inshape
# torch.save(stats_dict, args.model)

model_motion = vxm.networks.VxmDense(inshape=inshape).load(args.model_motion, device) # size = (64,64,64) inshape inside Vxm: (128, 128, 128)
model_motion.to(device)
model_motion.eval()

model_covis = vxm.networks.CoVisNet(inshape=inshape).load(args.model_covis, device) 
model_covis.to(device)
model_covis.eval()

# print(model)


warp_layer = SpatialTransformer(inshape).to('cuda') # size=(128,128,128)


for file_path in test_files:
    # if args.dataset == 'Clinical_echo':
    #     im_ED, im_ES, ED_myo, ES_myo, ED_epi, ES_epi, ED_endo, ES_endo = vxm.py.utils_clinical_echo.load_volfile_mat(file_path, add_batch_axis=True, add_feat_axis=True, registration=True)
    if args.dataset == 'Echo':
        im_ED, im_ES, ED_myo, ES_myo, ED_epi, ES_epi, ED_endo, ES_endo = vxm.py.utils_echo.load_volfile_mat(file_path, add_batch_axis=True, add_feat_axis=True, registration=True)

    if ndims == 3:
        input_ED = torch.from_numpy(im_ED).to(device).float().permute(0, 4, 1, 2, 3)
        input_ES = torch.from_numpy(im_ES).to(device).float().permute(0, 4, 1, 2, 3)
        input_ED_myo = torch.from_numpy(ED_myo).to(device).float().permute(0, 4, 1, 2, 3)

        # input_ED_epi = torch.from_numpy(ED_epi).to(device).float().permute(0, 4, 1, 2, 3)
        # input_ED_endo = torch.from_numpy(ED_endo).to(device).float().permute(0, 4, 1, 2, 3)

  

    im_ES_pred, dvf, mask_bgd = model_motion(input_ED, input_ES, registration=True)
    scoring_mask = model_covis(input_ES)


    
    ES_myo_pred = warp_layer(input_ED_myo, dvf.to(device))



    # resize and rescale dvf


    # print(logvar_map.shape)

    # if ndims == 3:
        # ES_epi_pred = warp_layer(input_ED_epi, dvf.to(device))
        # ES_endo_pred = warp_layer(input_ED_endo, dvf.to(device))
    
    if args.dataset == 'Echo':
        save_file = {
                "im_ED": im_ED[0, :, :, :, 0], 
                "im_ES": im_ES[0, :, :, :, 0], 
                "im_ES_pred": im_ES_pred.detach().cpu().numpy()[0,0], 
                "ED_myo": ED_myo[0, :, :, :, 0], 
                "ES_myo":ES_myo[0, :, :, :, 0], 
                "ES_myo_pred": ES_myo_pred.detach().cpu().numpy()[0,0], 
                "dvf": dvf.detach().cpu().numpy()[0], 
                "mask_bgd": mask_bgd.detach().cpu().numpy()[0,0],
                "scoring_mask": scoring_mask.detach().cpu().numpy()[0,0]
                }
  

    save_name = file_path.split('/')[-1]
    savemat(os.path.join(args.result_dir, save_name) ,save_file)

