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
os.environ['VXM_BACKEND'] = 'pytorch'
import voxelmorph as vxm  # nopep8

# wandb for tracking experiments
import wandb

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
parser.add_argument('--load-model', help='optional model file to initialize with')
parser.add_argument('--initial-epoch', type=int, default=0,
                    help='initial epoch number (default: 0)')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-4)')
parser.add_argument('--cudnn-nondet', action='store_true',
                    help='disable cudnn determinism - might slow down training')
parser.add_argument('--save_freq', type=int, default=20, 
                    help='save frequency')


# network architecture parameters
parser.add_argument('--enc', type=int, nargs='+',
                    help='list of unet encoder filters (default: 16 32 32 32)')
parser.add_argument('--dec', type=int, nargs='+',
                    help='list of unet decorder filters (default: 32 32 32 32 32 16 16)')
parser.add_argument('--int-steps', type=int, default=7,
                    help='number of integration steps (default: 7)')
parser.add_argument('--int-downsize', type=int, default=2,
                    help='flow downsample factor for integration (default: 2)')
parser.add_argument('--bidir', action='store_true', help='enable bidirectional cost function')

# loss hyperparameters
parser.add_argument('--image-loss', required=True,
                    help='image reconstruction loss - can be mse or ncc (default: mse)')
parser.add_argument('--lambda', type=float, dest='weight', default=0.01,
                    help='weight of deformation loss (default: 0.01)')

# gradient accumulation
parser.add_argument('--accumulation_steps', type=int, default=1, help='number of steps before backward and optimizer step')

# wandb run name
parser.add_argument('--wandb-name', type=str, required=True, help='name of wandb run')

args = parser.parse_args()



bidir = args.bidir

# load and prepare training data
# train_files = vxm.py.utils.read_file_list(args.img_list, prefix=args.img_prefix,
#                                           suffix=args.img_suffix)

# train_files = glob.glob('../../Dataset/Echo/train/*.mat') + glob.glob('../../Dataset/Echo/val/*.mat')
assert args.dataset in ['Echo', 'CAMUS', 'ACDC', 'Clinical_echo']
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
    generator = vxm.generators_echo.scan_to_scan_echo_3frame(
        train_files, batch_size=reduced_batch_size, bidir=args.bidir, add_feat_axis=add_feat_axis)
    
# extract shape from sampled input
inshape = next(generator)[0][0].shape[1:-1]

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

# unet architecture
enc_nf = args.enc if args.enc else [16, 32, 32, 32]
dec_nf = args.dec if args.dec else [32, 32, 32, 32, 32, 16, 16]

if args.load_model:
    # load initial model (if specified)
    model = vxm.networks.VxmDense.load(args.load_model, device)
else:
    # otherwise configure new model
    model = vxm.networks.VxmDense(
        inshape=inshape,
        nb_unet_features=[enc_nf, dec_nf],
        bidir=bidir,
        int_steps=args.int_steps,
        int_downsize=args.int_downsize
    )

if nb_gpus > 1:
    # use multiple GPUs via DataParallel
    model = torch.nn.DataParallel(model)
    model.save = model.module.save

# prepare the model for training and send to device
model.to(device)
model.train()

# set optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

# prepare image loss
if args.image_loss == 'ncc':
    image_loss_func = vxm.losses.NCC().loss
elif args.image_loss == 'mse':
    image_loss_func = vxm.losses.MSE().loss


else:
    raise ValueError('Image loss should be "mse" or "ncc", but found "%s"' % args.image_loss)

# need two image loss functions if bidirectional
if bidir:
    losses = [image_loss_func, image_loss_func]
    weights = [0.5, 0.5]
else:
    losses = [image_loss_func]
    weights = [1]

# prepare deformation loss
losses += [vxm.losses.Grad('l2', loss_mult=args.int_downsize).loss]
weights += [args.weight]


# wandb tracking experiments
run = wandb.init(
    # set the wandb project where this run will be logged
    project="cardiac_motion_baselines",
    
    # track hyperparameters and run metadata
    config={
    "learning_rate": args.lr,
    "architecture": "UNet",
    "dataset": args.dataset,
    "epochs": args.epochs,
    "batch_size": args.batch_size,
    "steps_per_epoch": args.steps_per_epoch,
    "bidir": args.bidir,
    "enc_nf": enc_nf,
    "dec_nf": dec_nf,
    "image-loss": args.image_loss,
    "loss_weights": weights,
    "accumulation_steps": args.accumulation_steps
    },

    # have a run name
    name = args.wandb_name
)


# training loops
for epoch in range(args.initial_epoch, args.epochs):

    # save model checkpoint
    if epoch % args.save_freq == 0:
        # wandb.save(os.path.join(wandb.run.dir, '%04d.pt' % epoch))
        model.save(os.path.join(model_dir, '%04d.pt' % epoch))
        # artifact = wandb.Artifact('voxelmorph', type='model')
        # artifact.add_file(os.path.join(model_dir, '%04d.pt' % epoch))
        # run.log_artifact(artifact)

    epoch_loss = []
    epoch_total_loss = []
    epoch_step_time = []

    # track gpu memory for batch accumulation experiment

    for step in range(args.steps_per_epoch):

        step_start_time = time.time()

        # generate inputs (and true outputs) and convert them to tensors
        inputs_EDES, y_true_EDES, inputs_EDmid, y_true_EDmid, inputs_midES, y_true_midES = next(generator)

        inputs_EDES = [torch.from_numpy(d).to(device).float().permute(0, 4, 1, 2, 3) for d in inputs_EDES]
        y_true_EDES = [torch.from_numpy(d).to(device).float().permute(0, 4, 1, 2, 3) for d in y_true_EDES]

        inputs_EDmid = [torch.from_numpy(d).to(device).float().permute(0, 4, 1, 2, 3) for d in inputs_EDmid]
        y_true_EDmid = [torch.from_numpy(d).to(device).float().permute(0, 4, 1, 2, 3) for d in y_true_EDmid]

        inputs_midES = [torch.from_numpy(d).to(device).float().permute(0, 4, 1, 2, 3) for d in inputs_midES]
        y_true_midES = [torch.from_numpy(d).to(device).float().permute(0, 4, 1, 2, 3) for d in y_true_midES]
        
        # run inputs through the model to produce a warped image and flow field
        y_pred_EDES = model(*inputs_EDES)
        y_pred_EDmid = model(*inputs_EDmid)
        y_pred_midES = model(*inputs_midES)

        # organize the pred and true of EDES, EDmid, midES
        y_pred = [y_pred_EDES, y_pred_EDmid, y_pred_midES]
        y_true = [y_true_EDES, y_true_EDmid, y_true_midES]

        # calculate total loss
        loss = 0
        loss_list = []
        for n, loss_function in enumerate(losses):
            curr_loss_sum = 0
            for frame_idx in range(3):
                curr_loss_frame = loss_function(y_true[frame_idx][n], y_pred[frame_idx][n], y_pred[frame_idx][-1]) * weights[n] # always return the background mask
                curr_loss_sum += curr_loss_frame

            curr_loss = curr_loss_sum / 3.0
            loss_list.append(curr_loss.item())
            loss += curr_loss

        epoch_loss.append(loss_list)
        epoch_total_loss.append(loss.item())

        # backpropagate and optimize
        loss.backward()

        if (step + 1) % args.accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        # get compute time
        epoch_step_time.append(time.time() - step_start_time)

        # track gpu memory
        memory_allocated = torch.cuda.memory_allocated(torch.cuda.current_device())
        memory_reserved = torch.cuda.memory_reserved(torch.cuda.current_device())

        # print(f"GPU Memory Allocated: {memory_allocated / (1024**2):.2f} MB")
        # print(f"GPU Memory Reserved: {memory_reserved / (1024**2):.2f} MB")


    # print epoch info
    epoch_info = 'Epoch %d/%d' % (epoch + 1, args.epochs)
    time_info = '%.4f sec/step' % np.mean(epoch_step_time)
    losses_info = ', '.join(['%.4e' % f for f in np.mean(epoch_loss, axis=0)])
    loss_info = 'loss: %.4e  (%s)' % (np.mean(epoch_total_loss), losses_info)
    memory_info = 'GPU Memory Allocated: %.2f MB, Reserved: %.2f MB' % (memory_allocated / (1024**2), memory_reserved / (1024**2))
    print(' - '.join((epoch_info, time_info, loss_info, memory_info)), flush=True)

    # print(len(epoch_loss[0]))
    # print(len(inputs))
    # print(inputs[0].shape)
    # # print(inputs.shape)
    # print(len(y_true))
    # print(y_true[0].shape)
    # print(len(y_pred))
    # print(y_pred[0].shape)
    # print(y_pred[1].shape)
    # print(y_pred[2].shape)

    z_idx = 32
    src_im = wandb.Image(inputs_EDES[0].cpu().detach().numpy()[0, 0][:, :, z_idx])
    tgt_im = wandb.Image(inputs_EDES[1].cpu().detach().numpy()[0, 0][:, :, z_idx])
    pred_tgt = wandb.Image(y_pred_EDES[0].cpu().detach().numpy()[0, 0][:, :, z_idx])
    
    if args.bidir:
        pred_src = wandb.Image(y_pred_EDES[1].cpu().detach().numpy()[0, 0,][:, :, z_idx])

    mask_bgd = wandb.Image(y_pred_EDES[-1].cpu().detach().numpy()[0, 0][:, :, z_idx])
    # print(src_im.shape)
    # print(tgt_im.shape)
    # print(pred_tgt.shape)
    # print(pred_src.shape)


    # track using wandb
    if args.bidir:
        wandb.log({
                "loss": np.mean(epoch_total_loss), 
                "fwd": np.mean(epoch_loss, axis=0)[0], 
                "bwd": np.mean(epoch_loss, axis=0)[1], 
                "reg": np.mean(epoch_loss, axis=0)[2], 
                "src_im":src_im, 
                "tgt_im": tgt_im, 
                "pred_tgt": pred_tgt, 
                "pred_src": pred_src,
                "mask_bgd": mask_bgd
                })
    else:
                wandb.log({
                "loss": np.mean(epoch_total_loss), 
                "fwd": np.mean(epoch_loss, axis=0)[0], 
                "reg": np.mean(epoch_loss, axis=0)[1], 
                "src_im":src_im, 
                "tgt_im": tgt_im, 
                "pred_tgt": pred_tgt, 
                "mask_bgd": mask_bgd
                })

# final model save, only do the versioning at end of training
# wandb.save(os.path.join(wandb.run.dir, '%04d.pt' % args.epochs))
model.save(os.path.join(model_dir, '%04d.pt' % args.epochs))
artifact = wandb.Artifact('voxelmorph', type='model')
artifact.add_file(os.path.join(model_dir, '%04d.pt' % args.epochs))
run.log_artifact(artifact)
