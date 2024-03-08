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
import transmorph as tsm  # nopep8
import configs, configs_2D

# wandb for tracking experiments
import wandb

from adaframe import compute_scoring_mask

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

parser.add_argument('--initial-epoch', type=int, default=0,
                    help='initial epoch number (default: 0)')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-4)')
parser.add_argument('--cudnn-nondet', action='store_true',
                    help='disable cudnn determinism - might slow down training')
parser.add_argument('--save_freq', type=int, default=50, 
                    help='save frequency')



# network architecture parameters
parser.add_argument('--bidir', action='store_true', help='enable bidirectional cost function')
parser.add_argument('--int-steps', type=int, default=7,
                    help='number of integration steps (default: 7)')
parser.add_argument('--int-downsize', type=int, default=2,
                    help='flow downsample factor for integration (default: 2)')

# loss hyperparameters
parser.add_argument('--motion-loss-type', required=True,
                    help='image reconstruction loss - can be mse or ncc or gc or adp (default: mse)')

parser.add_argument('--lambda', type=float, dest='weight', default=0.01,
                    help='weight of deformation loss (default: 0.01)')
parser.add_argument('--a0', type=float, default=0.1)
parser.add_argument('--b0', type=float, default=10)



# gradient accumulation
parser.add_argument('--accumulation_steps', type=int, default=1, help='number of steps before backward and optimizer step')

# wandb run name
parser.add_argument('--wandb-name', type=str, required=True, help='name of wandb run')



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
        model_motion = tsm.networks.TransMorph.load(args.load_model, device)
    elif ndims == 2:
        model_motion = tsm.networks_2D.TransMorph.load(args.load_model, device)
else:
    # otherwise configure new model
    config = config_transmorph

    if ndims == 3:
        model_motion = tsm.networks.TransMorph(config, args.bidir)
    elif ndims == 2:
        model_motion = tsm.networks_2D.TransMorph(config, args.bidir)




# prepare the model for training and send to device
model_motion.to(device)
model_motion.train()



# set optimizer
optimizer_motion = torch.optim.Adam(model_motion.parameters(), lr=args.lr, weight_decay=0, amsgrad=True)


# prepare image loss
if args.motion_loss_type == 'mse':
    loss_func_motion = tsm.losses.MSE().loss
elif args.motion_loss_type == 'wmse':
    loss_func_motion = tsm.losses.WeightedMSE().loss
else:
    raise ValueError('Image loss should be "mse" or "ncc", but found "%s"' % args.image_loss)


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


# wandb tracking experiments
run = wandb.init(
    # set the wandb project where this run will be logged
    project="cvpr2024_covis_cmp",
    
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
    "motion_loss_weights": weights_motion,

    "accumulation_steps": args.accumulation_steps,

    "a0": args.a0,
    "b0": args.b0
    },

    # have a run name
    name = args.wandb_name
)



# training loops
for epoch in range(args.initial_epoch, args.epochs):

    adjust_learning_rate(optimizer_motion, epoch, args.epochs, args.lr)

    # save model checkpoint
    if epoch % args.save_freq == 0:
        # wandb.save(os.path.join(wandb.run.dir, '%04d.pt' % epoch))
        model_motion.save(os.path.join(model_dir, 'motion_%04d.pt' % epoch))

    epoch_motion_loss = []
    epoch_motion_total_loss = []
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

        # run motion estimator forward pass at time t
        y_pred = model_motion(*inputs)

        # compute motion loss at time t
        motion_loss = 0
        motion_loss_list = []

        covis_vis = [] # save the forward and backward pass of covisibility map for visualization

        for n, loss_function in enumerate(losses_motion):

            # when computing the data term for the motion loss, use the predicted co-visiblity map
            if n < len(losses_motion) - 1:
                scoring_mask, rho, a, b = compute_scoring_mask(y_pred[n], y_true[n], y_pred[-1], args.a0, args.b0) # (input_ES, input_ED)
                # print(logsigma_image.shape) # B x 1 x H x W x D
                covis_vis.append(scoring_mask)
                   
                
            else:
                scoring_mask = None # for the last grad loss
                


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

    epoch_step_time.append(time.time() - step_start_time)

    # track gpu memory
    memory_allocated = torch.cuda.memory_allocated(torch.cuda.current_device())
    memory_reserved = torch.cuda.memory_reserved(torch.cuda.current_device())


    # print epoch info
    epoch_info = 'Epoch %d/%d' % (epoch + 1, args.epochs)
    time_info = '%.4f sec/step' % np.mean(epoch_step_time)
    memory_info = 'GPU Memory Allocated: %.2f MB, Reserved: %.2f MB' % (memory_allocated / (1024**2), memory_reserved / (1024**2))

    losses_info_motion = ', '.join(['%.4e' % f for f in np.mean(epoch_motion_loss, axis=0)])
    loss_info_motion = 'motion loss: %.4e  (%s)' % (np.mean(epoch_motion_total_loss), losses_info_motion)
    print(' - '.join((epoch_info, time_info, loss_info_motion, memory_info)), flush=True)


        
    # wandb visualization 
    if len(inshape) == 3:
        z_idx = 32

        src_im = wandb.Image(inputs[0].cpu().detach().numpy()[0, 0][:, :, z_idx])
        tgt_im = wandb.Image(inputs[1].cpu().detach().numpy()[0, 0][:, :, z_idx])
        pred_tgt = wandb.Image(y_pred[0].cpu().detach().numpy()[0, 0][:, :, z_idx])

        pred_scoring_mask_fwd = wandb.Image(covis_vis[0].cpu().detach().numpy()[0, 0][:, :, z_idx])
        pred_rho = wandb.Image(rho.cpu().detach().numpy()[0, 0][:, :, z_idx])

        if args.bidir:
            pred_src = wandb.Image(y_pred[1].cpu().detach().numpy()[0, 0,][:, :, z_idx])
            pred_scoring_mask_bwd = wandb.Image(covis_vis[1].cpu().detach().numpy()[0, 0][:, :, z_idx])


        mask_bgd = wandb.Image(y_pred[-1].cpu().detach().numpy()[0, 0][:, :, z_idx])
        

    if len(inshape) == 2:
        src_im = wandb.Image(inputs[0].cpu().detach().numpy()[0, 0])
        tgt_im = wandb.Image(inputs[1].cpu().detach().numpy()[0, 0])
        pred_tgt = wandb.Image(y_pred[0].cpu().detach().numpy()[0, 0])
    
        pred_scoring_mask_fwd = wandb.Image(covis_vis[0].cpu().detach().numpy()[0, 0])
        pred_rho = wandb.Image(rho.cpu().detach().numpy()[0, 0])

        


        if args.bidir:
            pred_src = wandb.Image(y_pred[1].cpu().detach().numpy()[0, 0,])
            pred_scoring_mask_bwd = wandb.Image(covis_vis[1].cpu().detach().numpy()[0, 0])



        mask_bgd = wandb.Image(y_pred[-1].cpu().detach().numpy()[0, 0])



    # track using wandb
    log_dict = {"motion_loss": np.mean(epoch_motion_total_loss), 
                "motion_fwd": np.mean(epoch_motion_loss, axis=0)[0], 
                "reg": np.mean(epoch_motion_loss, axis=0)[-1], 
                "src_im":src_im, 
                "tgt_im": tgt_im, 
                "pred_tgt": pred_tgt, 
                "pred_scoring_mask_fwd": pred_scoring_mask_fwd,
                "pred_scoring_mask_fwd_mean": np.mean(covis_vis[0].cpu().detach().numpy()[0, 0]),
                "mask_bgd": mask_bgd,
                "a": a.cpu().detach().numpy()[0],
                "b": b.cpu().detach().numpy()[0],
                "pred_rho": pred_rho
                }

    if args.bidir:
        log_dict['motion_bwd'] = np.mean(epoch_motion_loss, axis=0)[1]
        log_dict['pred_src'] = pred_src
        log_dict['pred_scoring_mask_bwd'] = pred_scoring_mask_bwd
        log_dict['pred_scoring_mask_bwd_mean'] = np.mean(covis_vis[1].cpu().detach().numpy()[0, 0])


    wandb.log(log_dict)


# save everything for the last epoch
model_motion.save(os.path.join(model_dir, 'motion_%04d.pt' % args.epochs))
artifact_motion = wandb.Artifact('transmorph_motion', type='model')
artifact_motion.add_file(os.path.join(model_dir, 'motion_%04d.pt' % args.epochs))
run.log_artifact(artifact_motion)
