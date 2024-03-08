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
    parser.add_argument('--load-model-covis', help='optional variance model file to initialize with')
    parser.add_argument('--warm_start_epoch', type=int, default=10, 
                    help='number of epochs to warm start the network')
    parser.add_argument('--int-steps', type=int, default=7,
                    help='number of integration steps (default: 7)')
    parser.add_argument('--int-downsize', type=int, default=2,
                    help='flow downsample factor for integration (default: 2)')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-4)')

    # loss
    parser.add_argument('--motion-loss-type', required=True,
                        help='image reconstruction loss - can be mse or ncc or gc or adp (default: mse)')
    parser.add_argument('--covis-loss-type', required=True,
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


    # warm start
    parser.add_argument('--warm_start', action='store_true', help='warm start the network by first training 10 epochs for motion esimator and then 10 epochs for covis estimator')


    # momentum loss related
    parser.add_argument('--beta', type=float, dest='beta', default=0.1,
                    help='weight of variance regularization loss (default: 0.1)')
    parser.add_argument('--momentum_scaling', type=float, default=50, help='decay rate of momentum')


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

    enc_nf = [16, 32, 32, 32]
    dec_nf = [32, 32, 32, 32, 32, 16, 16]
    if args.load_model_covis:
        # load initial model (if specified)
        model_covis = networks.CoVisNet.load(args.load_model_covis, device)
    else:
        # otherwise configure new model
        model_covis = networks.CoVisNet(
            inshape=inshape,
            nb_unet_features=[enc_nf, dec_nf],
            bidir=False,
            int_steps=args.int_steps,
            int_downsize=args.int_downsize
        )
    
    model_covis.to(device)
    model_covis.train()

    optimizer_covis = torch.optim.Adam(model_covis.parameters(), lr=args.lr)


    # prepare the covis loss
    if args.covis_loss_type == 'coviswmse':
        # loss_func_variance = vxm.losses.betaNLL(args.beta).loss
        loss_func_covis = covis_losses_bank.CoVisWeightedMSE().loss

    # prepare scoring loss
    losses_covis = [loss_func_covis]
    weights_covis = [1]

    losses_covis += [covis_losses_bank.ScoringLoss().loss]
    weights_covis += [args.alpha]

    losses_covis += [covis_losses_bank.CoVisTVLoss(ndims = ndims).loss]
    weights_covis += [args.beta]

    # wandb tracking
    # wandb tracking experiments
    run = wandb.init(
        # set the wandb project where this run will be logged
        project="cvpr2024_covis_tv",
        
        # track hyperparameters and run metadata
        config={
        "learning_rate": opt['train']['optimizer']['lr'],
        "architecture": "DiffuseMorph",
        "dataset": args.dataset,
        "epochs": opt['train']['n_epoch'],
        "batch_size": args.batch_size,
        "steps_per_epoch": args.steps_per_epoch,
        "motion-loss": opt['model']['motion_loss_type'],
        "covis-loss": args.covis_loss_type,
        "loss_weights": opt['model']['loss_lambda'],
        "loss_gamma": opt['model']['loss_gamma'],
        "accumulation_steps": args.accumulation_steps,

        "covis_loss_weights": weights_covis,

        # warm start
        "warm_start": args.warm_start,
        "warm_start_epoch": args.warm_start_epoch,

        "momentum_scaling": args.momentum_scaling
        },

        # have a run name
        name = args.wandb_name
    )

    warm_start_epoch = args.warm_start_epoch if args.warm_start == True else 0

    # model
    diffusion = Model.create_model(opt)
    logger.info('Initial Model Finished')

    momentum_prev = 0
    momentum_curr = 0

    momentum_decay = 0.99
    momentum_tv = 0

    # Train
    if opt['phase'] == 'train':
        current_step = diffusion.begin_step
        current_epoch = diffusion.begin_epoch
        n_epoch = opt['train']['n_epoch']
        if opt['path']['resume_state']:
            logger.info('Resuming training from epoch: {}, iter: {}.'.format(current_epoch, current_step))

        while current_epoch < n_epoch + 2 * warm_start_epoch:
            
            # warm start flags for motion and covis 
            if args.warm_start == True:
                if current_epoch < warm_start_epoch:
                    flag_motion = True
                    flag_covis = False
                elif current_epoch >= warm_start_epoch and current_epoch < 2 * warm_start_epoch:
                    flag_motion = False
                    flag_covis = True
                else:
                    flag_motion = True
                    flag_covis = True
            else:
                flag_motion = True
                flag_covis = True



            epoch_covis_loss = []
            epoch_covis_total_loss = []

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

                if flag_motion == True:

                    if flag_covis == False:
                        # first few epochs the variance estimator is not working
                        scoring_mask = torch.ones_like(y_true[0])
                    else:
                        # run forward pass from predictive covis model
                        scoring_mask = model_covis(y_true[0]) # (input_ES, input_ED)
                    


                    # convert to DiffuseMorph format
                    train_data = {'M': inputs[0], 'F': y_true[0], 'scoring_mask': scoring_mask}

                    current_step += 1

                    diffusion.feed_data_2frame(train_data)

                    # print(current_step)

                    # diffusion.optimize_parameters_2frame()

                    diffusion.optimize_parameters_2frame_grad_accum(idx_step=istep, grad_accum_step=args.accumulation_steps)


                    # loss residual
                    logs, img_logs = diffusion.get_current_log()
                    residuals = logs['l_sim']/args.lambda_L


                    weight_decay = np.min([1 - 1/(current_epoch+1), momentum_decay])

                    # print(residuals)
                    # mu_t = residuals

                    # momentum_curr = weight_decay * momentum_prev + (1 - weight_decay) * args.beta * np.cos(3.14 * args.momentum_scaling * mu_t)
                    # momentum_prev = momentum_curr

                    # weights_covis[-1] = momentum_curr

                #########################################################
                # run the forward pass of covis estimator from t to t+1
                #########################################################
                if flag_covis == True:
                    # run motion estimator at t+1
                    y_pred = diffusion.forward_2frame()
                    # print('y_pred shape: ', y_pred.shape)

                    # compute co-visbility loss
                    covis_loss = 0
                    covis_loss_list = []

                    # covis_vis = [] # save the forward and backward pass of covisibility map for visualization
                    # num_directions = 1 if args.bidir == False else 2

                    # for idx_bidir in range(num_directions):
                    scoring_mask = model_covis(y_true[0])
                        # covis_vis.append(scoring_mask)

                    weight_decay = np.min([1 - 1/(current_epoch+1), momentum_decay])

                    for n, loss_function in enumerate(losses_covis):

                        # for idx_bidir in range(num_directions):
                        curr_loss = loss_function(y_true[0], y_pred[0], scoring_mask, y_pred[-1]) * weights_covis[n]
                        
                        if ndims == 3:
                            total_tv = compute_tv(scoring_mask.cpu().detach().numpy()[0, 0])
                        elif ndims == 2:
                            total_tv = compute_tv_2d(scoring_mask.cpu().detach().numpy()[0, 0])
                            
                        momentum_tv = weight_decay * momentum_tv + (1 - weight_decay) * total_tv

                        covis_loss_list.append(curr_loss.item())
                        covis_loss += curr_loss
                    
                    # backpropagate and optimize t -> t+1
                    covis_loss.backward()

                    if (istep + 1) % args.accumulation_steps == 0:
                        optimizer_covis.step()
                        optimizer_covis.zero_grad()

                    epoch_covis_loss.append(covis_loss_list)
                    epoch_covis_total_loss.append(covis_loss.item())
                    



            # save for each epoch
            if flag_motion == True and flag_covis == True:
                logs, img_logs = diffusion.get_current_log()
                # logs content:  OrderedDict([('l_pix', xxx), ('l_sim', xxx), ('l_smt', xxx), ('l_tot', xxx)])
                # img_logs content: ['x_recon', 'out_M']
                # print(img_logs.keys()) # x_recon, out_M
                # print(img_logs['x_recon'].shape) # 1 x 1 x 64 x 64 x 64
                # print(img_logs['out_M'].shape) # 1 x 1 x 64 x 64 x 64

                if ndims == 3:
                    z_idx = 32
                    src_im = wandb.Image(inputs[0].cpu().detach().numpy()[0, 0][:, :, z_idx])
                    tgt_im = wandb.Image(inputs[1].cpu().detach().numpy()[0, 0][:, :, z_idx])
                    pred_tgt = wandb.Image(img_logs['out_M'].cpu().detach().numpy()[0, 0][:, :, z_idx])
                    x_recon = wandb.Image(img_logs['x_recon'].cpu().detach().numpy()[0, 0][:, :, z_idx])
                    mask_bgd = wandb.Image(img_logs['mask_bgd'].cpu().detach().numpy()[0, 0][:, :, z_idx])
                    pred_scoring_mask = wandb.Image(scoring_mask.cpu().detach().numpy()[0, 0][:, :, z_idx])

                elif ndims == 2:
                    src_im = wandb.Image(inputs[0].cpu().detach().numpy()[0, 0])
                    tgt_im = wandb.Image(inputs[1].cpu().detach().numpy()[0, 0])
                    pred_tgt = wandb.Image(img_logs['out_M'].cpu().detach().numpy()[0, 0])
                    x_recon = wandb.Image(img_logs['x_recon'].cpu().detach().numpy()[0, 0])
                    mask_bgd = wandb.Image(img_logs['mask_bgd'].cpu().detach().numpy()[0, 0])
                    pred_scoring_mask = wandb.Image(scoring_mask.cpu().detach().numpy()[0, 0])

                logs['src_im'] =  src_im
                logs['tgt_im'] = tgt_im
                logs['pred_tgt'] = pred_tgt
                logs['x_recon'] = x_recon
                logs['mask_bgd'] = mask_bgd

                # covis logging
                logs["covis_loss"] = np.mean(epoch_covis_total_loss)

                # print(epoch_covis_loss)
                # print(np.array(epoch_covis_loss).shape)
                # print(np.mean(epoch_covis_loss))
                logs["covis_data"] = np.mean(epoch_covis_loss, axis=0)[0]
                logs["covis_reg"] = np.mean(epoch_covis_loss, axis=0)[1]
                logs["covis_tv"] = np.mean(epoch_covis_loss, axis=0)[2]

                logs["pred_scoring_mask"] = pred_scoring_mask
                logs["pred_scoring_mask_mean"] = np.mean(scoring_mask.cpu().detach().numpy()[0, 0])

                # motion logging
                logs["momentum_tv"] = momentum_tv
                logs["weight_beta"] = weights_covis[-1]


                print(logs)

                wandb.log(logs)

            

            # track gpu memory
            memory_allocated = torch.cuda.memory_allocated(torch.cuda.current_device())
            memory_reserved = torch.cuda.memory_reserved(torch.cuda.current_device())

            memory_info = 'GPU Memory Allocated: %.2f MB, Reserved: %.2f MB' % (memory_allocated / (1024**2), memory_reserved / (1024**2))
            print(memory_info)




                    # visualizer.print_current_errors(current_epoch, istep + 1, training_iters, logs, t, 'Train')
                    # visualizer.plot_current_errors(current_epoch, (istep + 1) / float(training_iters), logs)

                # # validation
                # if (istep + 1) % opt['train']['val_freq'] == 0:
                #     result_path = '{}/{}'.format(opt['path']['results'], current_epoch)
                #     os.makedirs(result_path, exist_ok=True)

                #     diffusion.test_registration(continous=False)
                #     visuals = diffusion.get_current_visuals()
                #     visualizer.display_current_results(visuals, current_epoch, True)

            if current_epoch % opt['train']['save_checkpoint_epoch'] == 0:
                logger.info('Saving models and training states.')
                diffusion.save_network(current_epoch, current_step)
                model_covis.save(os.path.join(args.model_dir, 'covis_%04d.pt' % current_epoch))

        # save model

        logger.info('End of training.')
