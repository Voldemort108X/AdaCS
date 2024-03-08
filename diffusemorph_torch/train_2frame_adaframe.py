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

import model.networks as networks
import model.losses_covis as covis_losses_bank

from helper import *
from adaframe import compute_scoring_mask

import glob
import wandb

def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy.astype('uint8'))
    image_pil.save(image_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'test'],
                        help='Run either train(training) or val(generation)', default='train')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)

    # model parameters
    parser.add_argument('--save_freq', type=int, default=50, 
                    help='save frequency')
    
    parser.add_argument('--int-steps', type=int, default=7,
                    help='number of integration steps (default: 7)')
    parser.add_argument('--int-downsize', type=int, default=2,
                    help='flow downsample factor for integration (default: 2)')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-4)')

    # loss
    parser.add_argument('--motion-loss-type', required=True,
                        help='image reconstruction loss - can be mse or ncc or gc or adp (default: mse)')

    parser.add_argument('--alpha', type=float, dest='alpha', default=0.01,
                    help='weight of scoring regularization loss (default: 0.01)')
    parser.add_argument('--lambda_L', type=float, default=20, help='weight for image similarity loss (default: 20)')
    parser.add_argument('--gamma', type=float, default=1, help='weight for regularization loss (default: 0.01)')

    # data organization parameters
    parser.add_argument('--dataset', required=True, help='Name of the dataset')
    parser.add_argument('--model-dir', required=True,
                    help='model output directory (default: models)')
    
    # training parameters
    parser.add_argument('--batch-size', type=int, default=1, help='batch size (default: 1)')
    parser.add_argument('--steps-per-epoch', type=int, default=100,
                    help='frequency of model saves (default: 100)')
    parser.add_argument('--epoch', type=int, default=150, help='learning rate (default: 1e-4)')
    
    # gradient accumulation
    parser.add_argument('--accumulation_steps', type=int, default=1, help='number of steps before backward and optimizer step')

    parser.add_argument('-debug', '-d', action='store_true')

    # wandb run name
    parser.add_argument('--wandb-name', type=str, required=True, help='name of wandb run')

    parser.add_argument('--a0', type=float, default=0.1)
    parser.add_argument('--b0', type=float, default=10)

    # device
    device = 'cuda'
    

    # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)

    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    Logger.setup_logger(None, args.model_dir, 'train', level=logging.INFO, screen=True)

    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))

    # dataset
    assert args.dataset in ['Echo', 'CAMUS', 'ACDC']
    print(os.listdir('../../'))
    train_files = glob.glob(os.path.join('../../Dataset/', args.dataset, 'train/*.mat')) + glob.glob(os.path.join('../../Dataset', args.dataset, 'val/*.mat'))
    assert len(train_files) > 0, 'Could not find any training data.'

    # compute the real batch size needed
    reduced_batch_size = int(args.batch_size / args.accumulation_steps)
    if args.dataset == 'Echo':
        generator = data_gen.generators_echo.scan_to_scan_echo(
        train_files, batch_size=reduced_batch_size, bidir=False, add_feat_axis=True)
    elif args.dataset == 'CAMUS' or args.dataset == 'ACDC':
        generator = data_gen.generators_2D.scan_to_scan_2D(
        train_files, batch_size=reduced_batch_size, bidir=False, add_feat_axis=True)

    # get model save dir
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    opt['path']['checkpoint'] = args.model_dir

    # get datashape
    inshape = next(generator)[0][0].shape[1:-1]
    ndims = len(inshape)

    if ndims == 3:
        opt['model']['diffusion']['image_size'] = inshape
    elif ndims == 2:
        assert inshape[0] == inshape[1]
        opt['model']['diffusion']['image_size'] = inshape[0]

    opt['train']['n_epoch'] = args.epoch

    logger.info('Initial Dataset Finished')


    opt['model']['motion_loss_type'] = args.motion_loss_type

    # loss weights
    opt['model']['loss_lambda'] = args.lambda_L
    opt['model']['loss_gamma'] = args.gamma


    
    # wandb tracking
    # wandb tracking experiments
    run = wandb.init(
        # set the wandb project where this run will be logged
        project="cvpr2024_covis_cmp",
        
        # track hyperparameters and run metadata
        config={
        "learning_rate": opt['train']['optimizer']['lr'],
        "architecture": "DiffuseMorph",
        "dataset": args.dataset,
        "epochs": opt['train']['n_epoch'],
        "batch_size": args.batch_size,
        "steps_per_epoch": args.steps_per_epoch,
        "motion-loss": opt['model']['motion_loss_type'],

        "loss_weights": opt['model']['loss_lambda'],
        "loss_gamma": opt['model']['loss_gamma'],
        "accumulation_steps": args.accumulation_steps,

        "a0": args.a0,
        "b0": args.b0

        },

        # have a run name
        name = args.wandb_name
    )


    # model
    diffusion = Model.create_model(opt)
    logger.info('Initial Model Finished')

    momentum_decay = 0.99
    momentum_tv = 0

    # Train
    if opt['phase'] == 'train':
        current_step = diffusion.begin_step
        current_epoch = diffusion.begin_epoch
        n_epoch = opt['train']['n_epoch']
        if opt['path']['resume_state']:
            logger.info('Resuming training from epoch: {}, iter: {}.'.format(current_epoch, current_step))

        while current_epoch < n_epoch:
            

            current_epoch += 1
            
            # for istep, train_data in enumerate(train_loader):
            for istep in range(args.steps_per_epoch):

                iter_start_time = time.time()

                # generate inputs (and true outputs) and convert them to tensors
                inputs, y_true = next(generator)

                if ndims == 3:
                    inputs = [torch.from_numpy(d).to(device).float().permute(0, 4, 1, 2, 3) for d in inputs]
                    y_true = [torch.from_numpy(d).to(device).float().permute(0, 4, 1, 2, 3) for d in y_true]
                elif ndims == 2:
                    inputs = [torch.from_numpy(d).to(device).float().permute(0, 3, 1, 2) for d in inputs]
                    y_true = [torch.from_numpy(d).to(device).float().permute(0, 3, 1, 2) for d in y_true]


                #########################################################
                # run the forward pass of motion estimator from t to t+1
                #########################################################

                # convert to DiffuseMorph format
                train_data = {'M': inputs[0], 'F': y_true[0], 'scoring_mask': torch.ones_like(y_true[0])}

                diffusion.feed_data_2frame(train_data)
                y_pred, mask_bgd = diffusion.forward_2frame_adaframe()

                scoring_mask, rho, a, b = compute_scoring_mask(y_pred, y_true[0], mask_bgd, args.a0, args.b0)

                train_data = {'M': inputs[0], 'F': y_true[0], 'scoring_mask': scoring_mask}


                current_step += 1

                diffusion.feed_data_2frame(train_data)

                # print(current_step)

                # diffusion.optimize_parameters_2frame()

                diffusion.optimize_parameters_2frame_grad_accum(idx_step=istep, grad_accum_step=args.accumulation_steps)


            logs, img_logs = diffusion.get_current_log()

            if ndims == 3:
                z_idx = 32
                src_im = wandb.Image(inputs[0].cpu().detach().numpy()[0, 0][:, :, z_idx])
                tgt_im = wandb.Image(inputs[1].cpu().detach().numpy()[0, 0][:, :, z_idx])
                pred_tgt = wandb.Image(img_logs['out_M'].cpu().detach().numpy()[0, 0][:, :, z_idx])
                x_recon = wandb.Image(img_logs['x_recon'].cpu().detach().numpy()[0, 0][:, :, z_idx])
                mask_bgd = wandb.Image(img_logs['mask_bgd'].cpu().detach().numpy()[0, 0][:, :, z_idx])

                pred_scoring_mask = wandb.Image(scoring_mask.cpu().detach().numpy()[0, 0][:, :, z_idx])
                pred_rho = wandb.Image(rho.cpu().detach().numpy()[0, 0][:, :, z_idx])

            elif ndims == 2:
                src_im = wandb.Image(inputs[0].cpu().detach().numpy()[0, 0])
                tgt_im = wandb.Image(inputs[1].cpu().detach().numpy()[0, 0])
                pred_tgt = wandb.Image(img_logs['out_M'].cpu().detach().numpy()[0, 0])
                x_recon = wandb.Image(img_logs['x_recon'].cpu().detach().numpy()[0, 0])
                mask_bgd = wandb.Image(img_logs['mask_bgd'].cpu().detach().numpy()[0, 0])

                pred_scoring_mask = wandb.Image(scoring_mask.cpu().detach().numpy()[0, 0])
                pred_rho = wandb.Image(rho.cpu().detach().numpy()[0, 0])

            logs['src_im'] =  src_im
            logs['tgt_im'] = tgt_im
            logs['pred_tgt'] = pred_tgt
            logs['x_recon'] = x_recon
            logs['mask_bgd'] = mask_bgd


            logs["pred_scoring_mask"] = pred_scoring_mask
            logs["pred_scoring_mask_mean"] = np.mean(scoring_mask.cpu().detach().numpy()[0, 0])
            logs["pred_rho"] = pred_rho

            logs["a"] = a.cpu().detach().numpy()[0]
            logs["b"] = b.cpu().detach().numpy()[0]




            print(logs)

            wandb.log(logs)

            

        # track gpu memory
        memory_allocated = torch.cuda.memory_allocated(torch.cuda.current_device())
        memory_reserved = torch.cuda.memory_reserved(torch.cuda.current_device())

        memory_info = 'GPU Memory Allocated: %.2f MB, Reserved: %.2f MB' % (memory_allocated / (1024**2), memory_reserved / (1024**2))
        print(memory_info)




        if current_epoch % opt['train']['save_checkpoint_epoch'] == 0:
            logger.info('Saving models and training states.')
            diffusion.save_network(current_epoch, current_step)
    

    # save model

    logger.info('End of training.')
