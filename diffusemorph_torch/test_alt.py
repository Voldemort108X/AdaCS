import torch
# import data as Data
import data_generators as data_gen
import model as Model
import argparse
import logging
import core.logger as Logger
import core.metrics as Metrics
import os
from math import *
import time
from PIL import Image
import numpy as np
import torch.nn.functional as F
import glob
import wandb
from helper import *
from scipy.io import loadmat, savemat
from model.deformation_net_2D import SpatialTransformer # SpatialTransformer is the same in deformation_net_2D or 3D

from model.networks import CoVisNet

def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy.astype('uint8'))
    image_pil.save(image_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # diffusemorph specific 
    parser.add_argument('-c', '--config', type=str, default='config/.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'test'],
                        help='Run either train(training) or val(generation)', default='train')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')

    parser.add_argument('--dataset', required=True, help='Name of the dataset')
    parser.add_argument('--test-dir', required=True, help='path to all test files')
    parser.add_argument('--result-dir', required=True, help='where to save the result')
    parser.add_argument('--model-motion', required=True, help='pytorch model for nonlinear registration')
    parser.add_argument('--model-covis', required=True, help='pytorch model for segmentation')
    parser.add_argument('--inshape', required=True, nargs='+', type=int, help='input shape of the network e.g. (64, 64, 64) or (128, 128)')
    parser.add_argument('--image-loss', default='wmse',
                    help='image reconstruction loss - can be mse or ncc (default: mse)')

    # # loss
    # parser.add_argument('--image-loss', required=True,
    #                 help='image reconstruction loss - can be mse or ncc (default: mse)')

    # # data organization parameters
    
    # parser.add_argument('--model-dir', required=True,
    #                 help='model output directory (default: models)')
    
    # # training parameters
    # parser.add_argument('--batch-size', type=int, default=1, help='batch size (default: 1)')
    # parser.add_argument('--steps-per-epoch', type=int, default=100,
    #                 help='frequency of model saves (default: 100)')
    
    # # gradient accumulation
    # parser.add_argument('--accumulation_steps', type=int, default=1, help='number of steps before backward and optimizer step')

    # parser.add_argument('-debug', '-d', action='store_true')

    # # wandb run name
    # parser.add_argument('--wandb-name', type=str, required=True, help='name of wandb run')

    # device
    device = 'cuda'
    



    # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)

    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    # get dim info
    inshape = tuple(args.inshape)
    ndims = len(inshape) 

    # initialize the network
    opt['path']['resume_state'] = args.model_motion
    if ndims == 3:
        opt['model']['diffusion']['image_size'] = inshape
    elif ndims == 2:
        assert inshape[0] == inshape[1]
        opt['model']['diffusion']['image_size'] = inshape[0]

    opt['model']['motion_loss_type'] = args.image_loss
    # # set up logger
    # Logger.setup_logger('test', opt['path']['log'], 'test', level=logging.INFO)
    # logger = logging.getLogger('base')
    # logger.info(Logger.dict2str(opt))

    diffusion = Model.create_model(opt)

    # load covis model
    model_covis = CoVisNet(inshape=inshape).load(args.model_covis, device) 
    model_covis.to(device)
    model_covis.eval()


    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # Logger.setup_logger(None, args.model_dir, 'train', level=logging.INFO, screen=True)

    # logger = logging.getLogger('base')
    # logger.info(Logger.dict2str(opt))

    # get test dataset
    assert args.dataset in ['Echo', 'CAMUS', 'ACDC']
    test_files = glob.glob(os.path.join(args.test_dir, '*.mat'))
    assert len(test_files) > 0



    if ndims == 2:
        warp_layer = SpatialTransformer(inshape).to('cuda') # size=(128,128,128)
    

    for file_path in test_files:
        if args.dataset == 'Echo':
            im_ED, im_ES, ED_myo, ES_myo, ED_epi, ES_epi, ED_endo, ES_endo = data_gen.py.utils_echo.load_volfile_mat(file_path, add_batch_axis=True, add_feat_axis=True, registration=True)
        if args.dataset == 'CAMUS' or args.dataset == 'ACDC':
            im_ED, im_ES, ED_myo, ES_myo = data_gen.py.utils_2D.load_imfile_mat(file_path, add_batch_axis=True, add_feat_axis=True, registration=True)

        if ndims == 3:
            input_ED = torch.from_numpy(im_ED).to(device).float().permute(0, 4, 1, 2, 3)
            input_ES = torch.from_numpy(im_ES).to(device).float().permute(0, 4, 1, 2, 3)
            input_ED_myo = torch.from_numpy(ED_myo).to(device).float().permute(0, 4, 1, 2, 3)

            # input_ED_epi = torch.from_numpy(ED_epi).to(device).float().permute(0, 4, 1, 2, 3)
            # input_ED_endo = torch.from_numpy(ED_endo).to(device).float().permute(0, 4, 1, 2, 3)

        elif ndims == 2:
            input_ED = torch.from_numpy(im_ED).to(device).float().permute(0, 3, 1, 2)
            input_ES = torch.from_numpy(im_ES).to(device).float().permute(0, 3, 1, 2)
            input_ED_myo = torch.from_numpy(ED_myo).to(device).float().permute(0, 3, 1, 2)

        test_data = {'M': input_ED, 'F': input_ES, 'nS': torch.from_numpy(np.array(7)).to(device)}
        diffusion.feed_data_2frame(test_data)

        diffusion.test_registration()

        out_dict = diffusion.get_current_registration()

        # print(out_dict.keys())
        # print(out_dict['out_M'].shape) # 1x1x64x64x64
        # print(out_dict['flow'].shape) # 1x3x64x64x64

        im_ES_pred = out_dict['out_M'].numpy()[0,0]
        dvf = out_dict['flow'].numpy()[0]

        input_bank = torch.cat((out_dict['out_M'].to(device), input_ES), dim=1)
        scoring_mask = model_covis(input_bank)
        # scoring_mask = model_covis(input_ES)

        # print(im_ES_pred.shape)
        # print(dvf.shape)


        if args.dataset == 'Echo':
            # get target shape from test_noresize
            test_noresize_path = file_path.replace('test', 'test_noresize')
            file_noresize = loadmat(test_noresize_path)
            tgt_shape = [file_noresize['imshape'][0,j] for j in range(3)]

            # resize and rescale dvf
            dvf = dvf_resize(dvf, tgt_shape)

            # get im_ED, im_ES, ED_myo, ES_myo in original shape
            im_ED, im_ES, ED_myo, ES_myo = load_test_echo_noresize(file_noresize)

            # warp the im_ED and ED_myo to get the warped images
            warp_layer = SpatialTransformer(tgt_shape)
            input_ED_myo = torch.tensor(ED_myo, dtype=torch.get_default_dtype())[None,None,:,:,:]
            input_ED = torch.tensor(im_ED, dtype=torch.get_default_dtype())[None,None,:,:,:]
            input_dvf = torch.tensor(dvf, dtype=torch.get_default_dtype())[None,:,:,:,:]
            ES_myo_pred = warp_layer(input_ED_myo, input_dvf).detach().numpy()[0,0,:,:,:]
            im_ES_pred = warp_layer(input_ED, input_dvf).detach().numpy()[0,0,:,:,:]

            mask_bgd = (im_ED!=0) + (im_ES!=0)

        elif args.dataset == 'ACDC' or args.dataset == 'CAMUS':
            input_dvf = torch.tensor(dvf, dtype=torch.get_default_dtype())[None,:,:,:]

            ES_myo_pred = warp_layer(input_ED_myo, input_dvf.to(device))

            mask_bgd = (im_ED!=0) + (im_ES!=0)

        if args.dataset == 'Echo':
            save_file = {
                    "im_ED": im_ED, 
                    "im_ES": im_ES, 
                    "im_ES_pred": im_ES_pred, 
                    "ED_myo": ED_myo, 
                    "ES_myo":ES_myo, 
                    "ES_myo_pred": ES_myo_pred, 
                    "dvf": dvf, 
                    "mask_bgd": mask_bgd,
                    "scoring_mask": scoring_mask.detach().cpu().numpy()[0,0]
                    }
            
        elif ndims == 2:
            save_file = {
                    "im_ED": im_ED[0, :, :, 0], 
                    "im_ES": im_ES[0, :, :, 0], 
                    "im_ES_pred": im_ES_pred, 
                    "ED_myo": ED_myo[0, :, :, 0], 
                    "ES_myo":ES_myo[0, :, :, 0], 
                    "ES_myo_pred": ES_myo_pred.detach().cpu().numpy()[0, 0], 
                    "dvf": dvf, 
                    "mask_bgd": mask_bgd,
                    "scoring_mask": scoring_mask.detach().cpu().numpy()[0,0]
                    }

        save_name = file_path.split('/')[-1]
        savemat(os.path.join(args.result_dir, save_name) ,save_file)

