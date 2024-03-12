#!/usr/bin/env python


import os
import random
import argparse
import time
import numpy as np
import torch
import glob


# import voxelmorph with pytorch backend
os.environ['NEURITE_BACKEND'] = 'pytorch'
os.environ['TSM_BACKEND'] = 'pytorch'
import models.transmorph as tsm  # nopep8
import configs, configs_2D

# wandb for tracking experiments
import wandb

from helper import *

# parse the commandline
parser = argparse.ArgumentParser()

# data organization parameters
# parser.add_argument('--img-list', required=True, help='line-seperated list of training files')
parser.add_argument('--dataset', required=True, help='Name of the dataset')
parser.add_argument('--img-prefix', help='optional input image file prefix')
parser.add_argument('--img-suffix', help='optional input image file suffix')
parser.add_argument('--atlas', help='atlas filename (default: data/atlas_norm.npz)')
parser.add_argument('--model-dir', required=True,
                    help='model output directory (default: models)')
parser.add_argument('--multichannel', action='store_true',
                    help='specify that data has multiple channels')

# training parameters
parser.add_argument('--gpu', default='0', help='GPU ID number(s), comma-separated (default: 0)')
parser.add_argument('--batch-size', type=int, default=1, help='batch size (default: 1)')
parser.add_argument('--epochs', type=int, default=150,
                    help='number of training epochs (default: 150)')
parser.add_argument('--steps-per-epoch', type=int, default=100,
                    help='frequency of model saves (default: 100)')
parser.add_argument('--load-model-motion', help='optional motion model file to initialize with')
parser.add_argument('--load-model-covis', help='optional variance model file to initialize with')
parser.add_argument('--initial-epoch', type=int, default=0,
                    help='initial epoch number (default: 0)')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-4)')
parser.add_argument('--cudnn-nondet', action='store_true',
                    help='disable cudnn determinism - might slow down training')
parser.add_argument('--save_freq', type=int, default=50, 
                    help='save frequency')
parser.add_argument('--warm_start_epoch', type=int, default=10, 
                    help='number of epochs to warm start the network')


# network architecture parameters
parser.add_argument('--bidir', action='store_true', help='enable bidirectional cost function')
parser.add_argument('--int-steps', type=int, default=7,
                    help='number of integration steps (default: 7)')
parser.add_argument('--int-downsize', type=int, default=2,
                    help='flow downsample factor for integration (default: 2)')

# loss hyperparameters
parser.add_argument('--motion-loss-type', required=True,
                    help='image reconstruction loss - can be mse or ncc or gc or adp (default: mse)')
parser.add_argument('--covis-loss-type', required=True,
                    help='image reconstruction loss - can be mse or ncc or gc or adp (default: mse)')
parser.add_argument('--lambda', type=float, dest='weight', default=0.01,
                    help='weight of deformation loss (default: 0.01)')
parser.add_argument('--alpha', type=float, dest='alpha', default=0.01,
                    help='weight of scoring regularization loss (default: 0.01)')
parser.add_argument('--beta', type=float, dest='beta', default=0.1,
                    help='weight of variance regularization loss (default: 0.1)')

# gradient accumulation
parser.add_argument('--accumulation_steps', type=int, default=1, help='number of steps before backward and optimizer step')

# parser.add_argument('--momentum_window', type=int, default=10, help='number of steps to compute momentum')
parser.add_argument('--momentum_scaling', type=float, default=50, help='decay rate of momentum')

# wandb run name
parser.add_argument('--wandb-name', type=str, required=True, help='name of wandb run')

# flags for ablation experiments
parser.add_argument('--warm_start', action='store_true', help='warm start the network by first training 10 epochs for motion esimator and then 10 epochs for covis estimator')


args = parser.parse_args()

bidir = args.bidir

# helper function
def adjust_learning_rate(optimizer, epoch, MAX_EPOCHES, INIT_LR, power=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(INIT_LR * np.power( 1 - (epoch) / MAX_EPOCHES ,power),8)


# load and prepare training data
# train_files = vxm.py.utils.read_file_list(args.img_list, prefix=args.img_prefix,
#                                           suffix=args.img_suffix)

# train_files = glob.glob('../../Dataset/Echo/train/*.mat') + glob.glob('../../Dataset/Echo/val/*.mat')
assert args.dataset in ['Echo', 'CAMUS', 'ACDC']
print(os.listdir('../../'))
train_files = glob.glob(os.path.join('../../Dataset/', args.dataset, 'train/*.mat')) + glob.glob(os.path.join('../../Dataset', args.dataset, 'val/*.mat'))
assert len(train_files) > 0, 'Could not find any training data.'



# no need to append an extra feature axis if data is multichannel
add_feat_axis = not args.multichannel

# if args.atlas:
#     # scan-to-atlas generator
#     atlas = vxm.py.utils.load_volfile(args.atlas, np_var='vol',
#                                       add_batch_axis=True, add_feat_axis=add_feat_axis)
#     generator = vxm.generators.scan_to_atlas(train_files, atlas,
#                                              batch_size=args.batch_size, bidir=args.bidir,
#                                              add_feat_axis=add_feat_axis)
# else:
    # scan-to-scan generator
# compute the real batch size needed
reduced_batch_size = int(args.batch_size / args.accumulation_steps)
if args.dataset == 'Echo':
    generator = tsm.generators_echo.scan_to_scan_echo(
        train_files, batch_size=reduced_batch_size, bidir=args.bidir, add_feat_axis=add_feat_axis)
elif args.dataset == 'CAMUS' or args.dataset == 'ACDC':
    generator = tsm.generators_2D.scan_to_scan_2D(
        train_files, batch_size=reduced_batch_size, bidir=args.bidir, add_feat_axis=add_feat_axis)
    
# extract shape from sampled input
inshape = next(generator)[0][0].shape[1:-1]
ndims = len(inshape)

if ndims == 3:
    config_transmorph = configs.CONFIGS['TransMorph']
elif ndims == 2:
    config_transmorph = configs_2D.CONFIGS['TransMorph']
config_transmorph['img_size'] = inshape

# prepare model folder
model_dir = args.model_dir
os.makedirs(model_dir, exist_ok=True)

# device handling
gpus = args.gpu.split(',')
nb_gpus = len(gpus)
device = 'cuda'
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
assert np.mod(args.batch_size, nb_gpus) == 0, \
    'Batch size (%d) should be a multiple of the nr of gpus (%d)' % (args.batch_size, nb_gpus)

# enabling cudnn determinism appears to speed up training by a lot
torch.backends.cudnn.deterministic = not args.cudnn_nondet

if args.load_model_motion:
    # load initial model (if specified)
    if ndims == 3:
        model_motion = tsm.networks.TransMorph.load(args.load_model_motion, device)
    elif ndims == 2:
        model_motion = tsm.networks_2D.TransMorph.load(args.load_model_motion, device)
else:
    # otherwise configure new model
    config = config_transmorph

    if ndims == 3:
        model_motion = tsm.networks.TransMorph(config, args.bidir)
    elif ndims == 2:
        model_motion = tsm.networks_2D.TransMorph(config, args.bidir)

enc_nf = [16, 32, 32, 32]
dec_nf = [32, 32, 32, 32, 32, 16, 16]
if args.load_model_covis:
    # load initial model (if specified)
    model_covis = tsm.networks.CoVisNet.load(args.load_model_covis, device)
else:
    # otherwise configure new model
    model_covis = tsm.networks.CoVisNet(
        inshape=inshape,
        nb_unet_features=[enc_nf, dec_nf],
        bidir=bidir,
        int_steps=args.int_steps,
        int_downsize=args.int_downsize
    )


# prepare the model for training and send to device
model_motion.to(device)
model_motion.train()

model_covis.to(device)
model_covis.train()

# set optimizer
optimizer_motion = torch.optim.Adam(model_motion.parameters(), lr=args.lr, weight_decay=0, amsgrad=True)
optimizer_covis = torch.optim.Adam(model_covis.parameters(), lr=args.lr)

# prepare image loss
if args.motion_loss_type == 'mse':
    loss_func_motion = tsm.losses.MSE().loss
elif args.motion_loss_type == 'wmse':
    loss_func_motion = tsm.losses.WeightedMSE().loss
else:
    raise ValueError('Image loss should be "mse" or "ncc", but found "%s"' % args.image_loss)
# prepare the covis loss
if args.covis_loss_type == 'coviswmse':
    # loss_func_variance = vxm.losses.betaNLL(args.beta).loss
    loss_func_covis = tsm.losses.CoVisWeightedMSE().loss

# need two image loss functions if bidirectional
if bidir:
    losses_motion = [loss_func_motion, loss_func_motion]
    weights_motion = [0.5, 0.5]
else:
    losses_motion = [loss_func_motion]
    weights_motion = [1]

# prepare deformation loss
if ndims == 3:
    losses_motion += [tsm.losses.Grad('l2').loss]
elif ndims == 2:
    losses_motion += [tsm.losses.Grad_2d('l2').loss]
else:
    raise NotImplementedError('Only 2D and 3D supported')

weights_motion += [args.weight]

# prepare covis loss
losses_covis = [loss_func_covis]
weights_covis = [1]

# prepare scoring loss
losses_covis += [tsm.losses.ScoringLoss().loss]
weights_covis += [args.alpha]

losses_covis += [tsm.losses.CoVisTVLoss(ndims = ndims).loss]
weights_covis += [args.beta]


warm_start_epoch = args.warm_start_epoch if args.warm_start == True else 0

# wandb tracking experiments
run = wandb.init(
    # set the wandb project where this run will be logged
    project="cvpr2024_covis_tv",
    
    # track hyperparameters and run metadata
    config={
    "learning_rate": args.lr,
    "architecture": "TransUNet",
    "dataset": args.dataset,
    "epochs": args.epochs,
    "batch_size": args.batch_size,
    "steps_per_epoch": args.steps_per_epoch,
    "bidir": args.bidir,
    "motion-loss": args.motion_loss_type,
    "covis-loss": args.covis_loss_type,
    "motion_loss_weights": weights_motion,
    "covis_loss_weights": weights_covis,

    "accumulation_steps": args.accumulation_steps,

    "warm_start": args.warm_start,
    "warm_start_epoch": warm_start_epoch,

    "momentum_scaling": args.momentum_scaling
    },

    # have a run name
    name = args.wandb_name
)

# define the momentum of the residuals
momentum_prev = 0
momentum_curr = 0

momentum_decay = 0.99
momentum_tv = 0
# loss_tv_momentum = [0,0] if args.bidir == True else [0]
# loss_tv_momentum_no_past_grad = [0, 0] if args.bidir == True else [0]

# training loops
for epoch in range(args.initial_epoch, args.epochs + 2 * warm_start_epoch):

    # warm start flags for motion and covis 
    if args.warm_start == True:
        if epoch < warm_start_epoch:
            flag_motion = True
            flag_covis = False
        elif epoch >= warm_start_epoch and epoch < 2 * warm_start_epoch:
            flag_motion = False
            flag_covis = True
        else:
            flag_motion = True
            flag_covis = True
    else:
        flag_motion = True
        flag_covis = True

    adjust_learning_rate(optimizer_motion, epoch, args.epochs + 2 * warm_start_epoch, args.lr)

    # save model checkpoint
    if epoch % args.save_freq == 0:
        # wandb.save(os.path.join(wandb.run.dir, '%04d.pt' % epoch))
        model_motion.save(os.path.join(model_dir, 'motion_%04d.pt' % epoch))
        model_covis.save(os.path.join(model_dir, 'covis_%04d.pt' % epoch))

    epoch_motion_loss, epoch_covis_loss = [], []
    epoch_motion_total_loss, epoch_covis_total_loss = [], []
    epoch_step_time = []

    # track gpu memory for batch accumulation experiment

    for step in range(args.steps_per_epoch):

        step_start_time = time.time()

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

            # run motion estimator forward pass at time t
            y_pred = model_motion(*inputs)

            # compute motion loss at time t
            motion_loss = 0
            motion_loss_list = []

            for n, loss_function in enumerate(losses_motion):

                # when computing the data term for the motion loss, use the predicted co-visiblity map
                if n < len(losses_motion) - 1:

                    if flag_covis == False:
                        # first few epochs the variance estimator is not working
                        scoring_mask = torch.ones_like(y_true[n])
                    else:
                        # run forward pass from predictive covis model
                        scoring_mask = model_covis(y_true[n]) # (input_ES, input_ED)
                    # print(logsigma_image.shape) # B x 1 x H x W x D
                
                else:
                    scoring_mask = None # for the last grad loss
                

                        # print('image weight',image_weight.shape) # B x 1 x H x W x D


                # print(n)
                # print(y_true[n].shape)
                # print(y_pred[n].shape)
                # if scoring_mask == None:
                #     print(scoring_mask)
                # else:
                #     print(scoring_mask.shape)
                # print(y_pred[-1].shape)

                curr_loss = loss_function(y_true[n], y_pred[n], scoring_mask, y_pred[-1]) * weights_motion[n]
                
                motion_loss_list.append(curr_loss.item())
                motion_loss += curr_loss

            # backpropagate and optimize t -> t+1
            motion_loss.backward()

            if (step + 1) % args.accumulation_steps == 0:
                optimizer_motion.step()
                optimizer_motion.zero_grad()

            epoch_motion_loss.append(motion_loss_list)
            epoch_motion_total_loss.append(motion_loss.item())

            # update the momentum of the residuals
            weight_decay = np.min([1 - 1/(epoch+1), momentum_decay])

            num_directions = 1 if args.bidir == False else 2
            mu_t = 0 # current motion loss
            for idx_direction in range(num_directions):
                mu_t += (1/num_directions) * np.mean(np.mean(epoch_motion_loss, axis=0)[idx_direction])

            momentum_curr = weight_decay * momentum_prev + (1 - weight_decay) * args.beta * np.cos(3.14 * args.momentum_scaling * mu_t)
            momentum_prev = momentum_curr

            weights_covis[-1] = momentum_curr
            # momentum_curr = weight_decay * momentum_prev + (1 - weight_decay) * np.mean(np.mean(epoch_motion_loss, axis=0)[0])

            # momentum_diff = momentum_curr - momentum_prev
            # momentum_prev = momentum_curr


        #########################################################
        # run the forward pass of covis estimator from t to t+1
        #########################################################
        if flag_covis == True:

            # run motion estimator at t+1
            y_pred = model_motion(*inputs)

            # compute co-visbility loss
            covis_loss = 0
            covis_loss_list = []

            covis_vis = [] # save the forward and backward pass of covisibility map for visualization
            num_directions = 1 if args.bidir == False else 2

            for idx_bidir in range(num_directions):
                scoring_mask = model_covis(y_true[idx_bidir])
                covis_vis.append(scoring_mask)

            # weight_decay = np.min([1 - 1/(epoch+1), momentum_decay])

            for n, loss_function in enumerate(losses_covis):
                curr_loss = 0
                total_tv = 0

                for idx_bidir in range(num_directions):
                #     if n == 2:
                #         # curr_loss += (1/num_directions) * loss_function(y_true[idx_bidir], y_pred[idx_bidir], covis_vis[idx_bidir], y_pred[-1]) * weights_covis[n]
                #         # continue
                #         # compute the total variation of the covisibility map
                        
                #         loss_tv_momentum[idx_bidir] = weight_decay * loss_tv_momentum_no_past_grad[idx_bidir] + (1 - weight_decay) * loss_function(y_true[idx_bidir], y_pred[idx_bidir], covis_vis[idx_bidir], y_pred[-1]) 
                #         curr_loss += (1/num_directions) * loss_tv_momentum[idx_bidir] * weights_covis[n]

                #         loss_tv_momentum_no_past_grad[idx_bidir] = loss_tv_momentum[idx_bidir].item()
                #     else:
                    curr_loss += (1/num_directions) * loss_function(y_true[idx_bidir], y_pred[idx_bidir], covis_vis[idx_bidir], y_pred[-1]) * weights_covis[n]

                
                    if ndims == 3:
                        total_tv += (1/num_directions) * compute_tv(covis_vis[idx_bidir].cpu().detach().numpy()[0, 0])
                    elif ndims == 2:
                        total_tv += (1/num_directions) * compute_tv_2d(covis_vis[idx_bidir].cpu().detach().numpy()[0, 0])
                    
                momentum_tv = weight_decay * momentum_tv + (1 - weight_decay) * total_tv

                covis_loss_list.append(curr_loss.item())
                covis_loss += curr_loss
            
            # backpropagate and optimize t -> t+1
            covis_loss.backward()

            if (step + 1) % args.accumulation_steps == 0:
                optimizer_covis.step()
                optimizer_covis.zero_grad()

            epoch_covis_loss.append(covis_loss_list)
            epoch_covis_total_loss.append(covis_loss.item())

         # get compute time
        epoch_step_time.append(time.time() - step_start_time)

    # # when epoch ends, update the momentum of the residuals

    # if flag_motion == True and flag_covis == True:
    #     if args.mom_guide == True:

    #         if (epoch + 1) % args.momentum_window == 0:
                        
    #             momentum_diff = momentum_curr - momentum_prev
    #             momentum_prev = momentum_curr
    #             '''
    #                     velocity_diff = momentum_diff_curr - momentum_diff_prev

    #                     momentum_prev = momentum_curr
    #                     momentum_diff_past = momentum_diff_curr

    #                     momentum_diff_curr /= args.momentum_window
    #                     velocity_diff /= args.momentum_window

    #                     # bound the decay factor with 0.99 or 1.01
    #                     weight_factor_mom = np.exp(momentum_diff_curr * args.momentum_scaling)
    #                     weight_factor_vel = np.exp(-velocity_diff * args.momentum_scaling)

    #                     weight_factor = weight_factor_mom * weight_factor_vel
    #             '''
    #             momentum_diff /= args.momentum_window
    #             weight_factor = np.exp(momentum_diff * args.momentum_scaling)
    #             weight_factor = np.clip(weight_factor, 0.9, 1.1)

    #             weights_covis[1] = weights_covis[1] * weight_factor


    # track gpu memory
    memory_allocated = torch.cuda.memory_allocated(torch.cuda.current_device())
    memory_reserved = torch.cuda.memory_reserved(torch.cuda.current_device())


    # print epoch info
    epoch_info = 'Epoch %d/%d' % (epoch + 1, args.epochs)
    time_info = '%.4f sec/step' % np.mean(epoch_step_time)
    memory_info = 'GPU Memory Allocated: %.2f MB, Reserved: %.2f MB' % (memory_allocated / (1024**2), memory_reserved / (1024**2))

    # for motion info
    if flag_motion:
        losses_info_motion = ', '.join(['%.4e' % f for f in np.mean(epoch_motion_loss, axis=0)])
        loss_info_motion = 'motion loss: %.4e  (%s)' % (np.mean(epoch_motion_total_loss), losses_info_motion)
        print(' - '.join((epoch_info, time_info, loss_info_motion, memory_info)), flush=True)

    # for variance info
    if flag_covis:
        losses_info_covis = ', '.join(['%.4e' % f for f in np.mean(epoch_covis_loss, axis=0)])
        loss_info_covis = 'covis loss: %.4e  (%s)' % (np.mean(epoch_covis_total_loss), losses_info_covis)
        print(' - '.join((epoch_info, time_info, loss_info_covis, memory_info)), flush=True)
        # print(f"GPU Memory Allocated: {memory_allocated / (1024**2):.2f} MB")
        # print(f"GPU Memory Reserved: {memory_reserved / (1024**2):.2f} MB")



    if flag_motion and flag_covis:
        
        # wandb visualization 
        if len(inshape) == 3:
            z_idx = 32

            src_im = wandb.Image(inputs[0].cpu().detach().numpy()[0, 0][:, :, z_idx])
            tgt_im = wandb.Image(inputs[1].cpu().detach().numpy()[0, 0][:, :, z_idx])
            pred_tgt = wandb.Image(y_pred[0].cpu().detach().numpy()[0, 0][:, :, z_idx])

            pred_scoring_mask_fwd = wandb.Image(covis_vis[0].cpu().detach().numpy()[0, 0][:, :, z_idx])

            # track total variation stats
            # entropy_fwd = torch.distributions.Categorical(probs=covis_vis[0].cpu().detach().numpy()[0, 0].flatten()).entropy()
            tv_fwd = compute_tv(covis_vis[0].cpu().detach().numpy()[0, 0])
            weight_fwd = 1 - np.exp(-np.mean(epoch_motion_loss, axis=0)[0])

            
            if args.bidir:
                pred_src = wandb.Image(y_pred[1].cpu().detach().numpy()[0, 0,][:, :, z_idx])
                pred_scoring_mask_bwd = wandb.Image(covis_vis[1].cpu().detach().numpy()[0, 0][:, :, z_idx])

                # entropy_bwd = torch.distributions.Categorical(probs=covis_vis[1].cpu().detach().numpy()[0, 0].flatten()).entropy()
                tv_bwd = compute_tv(covis_vis[1].cpu().detach().numpy()[0, 0])
                weight_bwd = 1 - np.exp(-np.mean(epoch_motion_loss, axis=0)[1])

            mask_bgd = wandb.Image(y_pred[-1].cpu().detach().numpy()[0, 0][:, :, z_idx])
        

        if len(inshape) == 2:
            src_im = wandb.Image(inputs[0].cpu().detach().numpy()[0, 0])
            tgt_im = wandb.Image(inputs[1].cpu().detach().numpy()[0, 0])
            pred_tgt = wandb.Image(y_pred[0].cpu().detach().numpy()[0, 0])
            
            pred_scoring_mask_fwd = wandb.Image(covis_vis[0].cpu().detach().numpy()[0, 0])

            # track entropy stats
            # entropy_fwd = torch.distributions.Categorical(probs=covis_vis[0].cpu().detach().numpy()[0, 0].flatten()).entropy()
            tv_fwd = compute_tv_2d(covis_vis[0].cpu().detach().numpy()[0, 0])
            weight_fwd = 1 - np.exp(-np.mean(epoch_motion_loss, axis=0)[0])


            if args.bidir:
                pred_src = wandb.Image(y_pred[1].cpu().detach().numpy()[0, 0,])
                pred_scoring_mask_bwd = wandb.Image(covis_vis[1].cpu().detach().numpy()[0, 0])

                # entropy_bwd = torch.distributions.Categorical(probs=covis_vis[1].cpu().detach().numpy()[0, 0].flatten()).entropy()
                tv_bwd = compute_tv_2d(covis_vis[1].cpu().detach().numpy()[0, 0])
                weight_bwd = 1 - np.exp(-np.mean(epoch_motion_loss, axis=0)[1])


            mask_bgd = wandb.Image(y_pred[-1].cpu().detach().numpy()[0, 0])



        # track using wandb
        log_dict = {"motion_loss": np.mean(epoch_motion_total_loss), 
                    "covis_loss": np.mean(epoch_covis_total_loss),
                    "motion_fwd": np.mean(epoch_motion_loss, axis=0)[0], 
                    "covis_data": np.mean(epoch_covis_loss, axis=0)[0], 
                    "covis_reg": np.mean(epoch_covis_loss, axis=0)[1],
                    "covis_tv": np.mean(epoch_covis_loss, axis=0)[2],
                    "reg": np.mean(epoch_motion_loss, axis=0)[-1], 
                    "src_im":src_im, 
                    "tgt_im": tgt_im, 
                    "pred_tgt": pred_tgt, 
                    "pred_scoring_mask_fwd": pred_scoring_mask_fwd,
                    "pred_scoring_mask_fwd_mean": np.mean(covis_vis[0].cpu().detach().numpy()[0, 0]),
                    "mask_bgd": mask_bgd,

                    "tv_fwd": tv_fwd,
                    "weight_fwd": weight_fwd,

                    "momentum_tv": momentum_tv,
                    "weight_beta": weights_covis[-1]
                    # "momentum": momentum_curr,
                    # "momentum_diff": momentum_diff,
                    # "velocity_diff": velocity_diff,
                    # "weight_mom": weight_factor_mom,
                    # "weight_vel": weight_factor_vel,
                    # "covis_reg_weight": weights_covis[1]
                    }

        if args.bidir:
            log_dict['motion_bwd'] = np.mean(epoch_motion_loss, axis=0)[1]
            log_dict['pred_src'] = pred_src
            log_dict['pred_scoring_mask_bwd'] = pred_scoring_mask_bwd
            log_dict['pred_scoring_mask_bwd_mean'] = np.mean(covis_vis[1].cpu().detach().numpy()[0, 0])

            log_dict['tv_bwd'] = tv_bwd
            log_dict['weight_bwd'] = weight_bwd

        wandb.log(log_dict)


# save everything for the last epoch
model_motion.save(os.path.join(model_dir, 'motion_%04d.pt' % args.epochs))
artifact_motion = wandb.Artifact('transmorph_motion', type='model')
artifact_motion.add_file(os.path.join(model_dir, 'motion_%04d.pt' % args.epochs))
run.log_artifact(artifact_motion)

model_covis.save(os.path.join(model_dir, 'covis_%04d.pt' % args.epochs))
artifact_covis = wandb.Artifact('transmorph_covis', type='model')
artifact_covis.add_file(os.path.join(model_dir, 'covis_%04d.pt' % args.epochs))
run.log_artifact(artifact_covis)
