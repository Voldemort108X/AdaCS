import glob
import os
import sys
from argparse import ArgumentParser
from datetime import datetime

import numpy as np
import torch
import torch.utils.data as Data
import torch.nn.functional as F

import wandb

# from Functions import generate_grid, Dataset_epoch, Dataset_epoch_validation, transform_unit_flow_to_flow_cuda, \
#     generate_grid_unit

from miccai2021_model_2D import Miccai2021_LDR_conditional_laplacian_unit_disp_add_lvl1, \
    Miccai2021_LDR_conditional_laplacian_unit_disp_add_lvl2, Miccai2021_LDR_conditional_laplacian_unit_disp_add_lvl3, \
    SpatialTransform_unit, SpatialTransformNearest_unit, smoothloss, \
    neg_Jdet_loss, MSE

import data_generators as data_gen

parser = ArgumentParser()
parser.add_argument("--lr", type=float,
                    dest="lr", default=1e-4, help="learning rate")
parser.add_argument("--iteration_lvl1", type=int,
                    dest="iteration_lvl1", default=10001,
                    help="number of lvl1 iterations")
parser.add_argument("--iteration_lvl2", type=int,
                    dest="iteration_lvl2", default=10001,
                    help="number of lvl2 iterations")
parser.add_argument("--iteration_lvl3", type=int,
                    dest="iteration_lvl3", default=20001,
                    help="number of lvl3 iterations")
parser.add_argument("--antifold", type=float,
                    dest="antifold", default=0.,
                    help="Anti-fold loss: Disabled")
parser.add_argument("--checkpoint", type=int,
                    dest="checkpoint", default=1000,
                    help="frequency of saving models")
parser.add_argument("--start_channel", type=int,
                    dest="start_channel", default=64,  # default:8, 7 for stage
                    help="number of start channels")
# parser.add_argument("--datapath", type=str,
#                     dest="datapath",
#                     default='../Data/OASIS',
                    # help="data path for training images")
parser.add_argument("--freeze_step", type=int,
                    dest="freeze_step", default=1500,
                    help="Number of step to freeze the previous level")

#########################
# customized args parser
#########################
parser.add_argument('--dataset', required=True, help='Name of the dataset')

parser.add_argument('--model-dir', required=True,
                    help='model output directory (default: models)')
parser.add_argument('--batch-size', type=int, default=4,
                    help='input batch size for training (default: 4)')
parser.add_argument('--steps_per_epoch', type=int, default=100,
                    help='number of steps per epoch')

parser.add_argument('--range_flow', type=float, default=1)
parser.add_argument('--max_smooth', type=float, default=0)

parser.add_argument('--wandb-name', type=str, required=True, help='name of wandb run')
#########################

opt = parser.parse_args()

lr = opt.lr
start_channel = opt.start_channel
antifold = opt.antifold
n_checkpoint = opt.checkpoint
# datapath = opt.datapath
freeze_step = opt.freeze_step

iteration_lvl1 = opt.iteration_lvl1
iteration_lvl2 = opt.iteration_lvl2
iteration_lvl3 = opt.iteration_lvl3

range_flow = opt.range_flow
max_smooth = opt.max_smooth

# model_name = 'LDR_{}_NCC_unit_disp_add_fea64_reg01_10_2D_'.format(str(opt.dataset))
#########################################
# prepare our customized data generator
#########################################
train_files = glob.glob(os.path.join('../../Dataset/', opt.dataset, 'train/*.mat')) + glob.glob(os.path.join('../../Dataset', opt.dataset, 'val/*.mat'))

batch_size = opt.batch_size

if opt.dataset == 'CAMUS' or opt.dataset == 'ACDC':
    generator = data_gen.generators_2D.scan_to_scan_2D(
        train_files, batch_size=batch_size, bidir=False, add_feat_axis=True)

# inner_iter = int(len(train_files) // batch_size)
inner_iter = int(opt.steps_per_epoch)
#########################################

#########################################
# wandb tracking setup
#########################################
run = wandb.init(
        # set the wandb project where this run will be logged
        project="cardiac_motion_baselines",
        
        # track hyperparameters and run metadata
        config={
        "learning_rate": opt.lr,
        "architecture": "cLapIRN",
        "dataset": opt.dataset,
        "iteration_lvl1": opt.iteration_lvl1,
        "iteration_lvl2": opt.iteration_lvl2,
        "iteration_lvl3": opt.iteration_lvl3,
        "batch_size": opt.batch_size,
        "range_flow": opt.range_flow,
        "max_smooth": opt.max_smooth,
        },

        # have a run name
        name = opt.wandb_name
    )

model_dir = opt.model_dir

if not os.path.exists(model_dir):
    os.makedirs(model_dir)
    
if not os.path.exists(model_dir + '/loss'):
    os.makedirs(model_dir + '/loss')

def train_lvl1():
    print("Training lvl1...")
    model = Miccai2021_LDR_conditional_laplacian_unit_disp_add_lvl1(2, 2, start_channel, is_train=True,
                                                                    imgshape=imgshape_4,
                                                                    range_flow=range_flow).cuda()

    # loss_similarity = NCC(win=3)
    loss_similarity = MSE().loss
    loss_smooth = smoothloss
    # loss_Jdet = neg_Jdet_loss

    transform = SpatialTransform_unit().cuda()

    for param in transform.parameters():
        param.requires_grad = False
        param.volatile = True

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    lossall = np.zeros((3, iteration_lvl1 + 1))


    step = 0

    while step <= iteration_lvl1:
        for i_iter in range(inner_iter):
            inputs, y_true = next(generator)

            inputs = [torch.from_numpy(d).to('cuda').float().permute(0, 3, 1, 2) for d in inputs]
            y_true = [torch.from_numpy(d).to('cuda').float().permute(0, 3, 1, 2) for d in y_true]

            X, Y = inputs[0], y_true[0]

            # print('X shape', X.shape)
            # print('Y shape', Y.shape)

        # for X, Y in training_generator:
            X = X.squeeze(-1).cuda().float()
            Y = Y.squeeze(-1).cuda().float()

            print('X shape', X.shape)
            print('Y shape', Y.shape)

            reg_code = torch.rand(1, dtype=X.dtype, device=X.device).unsqueeze(dim=0)

            F_X_Y, X_Y, Y_4x, F_xy, _ = model(X, Y, reg_code)

            # print('F_X_Y shape', F_X_Y.shape) # dvf_fwd downsampled
            # print('X_Y shape', X_Y.shape) # y_pred downsampled
            # print('Y_4x shape', Y_4x.shape) # y_true downsampled
            # print('F_xy shape', F_xy.shape) # copy of F_X_Y

            loss_multiNCC = loss_similarity(X_Y, Y_4x)

            # F_X_Y_norm = transform_unit_flow_to_flow_cuda(F_X_Y.permute(0, 2, 3, 1).clone())

            # loss_Jacobian = loss_Jdet(F_X_Y_norm, grid_4)

            _, _, x, y = F_X_Y.shape
            norm_vector = torch.zeros((1, 2, 1, 1), dtype=F_X_Y.dtype, device=F_X_Y.device)
            norm_vector[0, 0, 0, 0] = (y - 1)
            norm_vector[0, 1, 0, 0] = (x - 1)
            loss_regulation = loss_smooth(F_X_Y * norm_vector)

            smo_weight = reg_code * max_smooth
            loss = loss_multiNCC + smo_weight * loss_regulation

            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients

            lossall[:, step] = np.array(
                [loss.item(), loss_multiNCC.item(), loss_regulation.item()])
            sys.stdout.write(
                "\r" + 'step "{0}" -> training loss "{1:.4f}" - sim_NCC "{2:4f}" -smo "{3:.4f} -reg_c "{4:.4f}"'.format(
                    step, loss.item(), loss_multiNCC.item(), loss_regulation.item(), reg_code[0].item()))
            sys.stdout.flush()

            # with lr 1e-3 + with bias
            if (step % n_checkpoint == 0):
                # modelname = model_dir + '/' + model_name + "stagelvl1_" + str(step) + '.pth'
                modelname = model_dir + "/stagelvl1_" + str(step) + '.pth'
                torch.save(model.state_dict(), modelname)
                np.save(model_dir + '/loss' + "/stagelvl1_" + str(step) + '.npy', lossall)

                #########################################
                # set up wandb tracking
                #########################################
                src_im = wandb.Image(inputs[0].cpu().detach().numpy()[0, 0])
                tgt_im = wandb.Image(inputs[1].cpu().detach().numpy()[0, 0])
                tgt_im_curr_lv = wandb.Image(Y_4x.cpu().detach().numpy()[0, 0])
                pred_tgt_curr_lv = wandb.Image(X_Y.cpu().detach().numpy()[0, 0])

                logs = {
                    "loss": loss.item(),
                    "sim_NCC": loss_multiNCC.item(),
                    "smo": loss_regulation.item(),
                    "reg_c": reg_code[0].item(),
                    "src_im": src_im,
                    "tgt_im": tgt_im,
                    "tgt_im_curr_lv": tgt_im_curr_lv,
                    "pred_tgt_curr_lv": pred_tgt_curr_lv,
                }

                wandb.log(logs)
                
            step += 1

            if step > iteration_lvl1:
                break
        print("one epoch pass")
    np.save(model_dir + '/loss' + '/stagelvl1.npy', lossall)


def train_lvl2():
    print("Training lvl2...")
    model_lvl1 = Miccai2021_LDR_conditional_laplacian_unit_disp_add_lvl1(2, 2, start_channel, is_train=True,
                                                                         imgshape=imgshape_4,
                                                                         range_flow=range_flow).cuda()

    model_path = sorted(glob.glob(model_dir + "/stagelvl1_?????.pth"))[-1]
    # model_path = sorted(glob.glob(model_dir + "/stagelvl1_?.pth"))[-1]

    model_lvl1.load_state_dict(torch.load(model_path))
    print("Loading weight for model_lvl1...", model_path)

    # Freeze model_lvl1 weight
    for param in model_lvl1.parameters():
        param.requires_grad = False

    model = Miccai2021_LDR_conditional_laplacian_unit_disp_add_lvl2(2, 2, start_channel, is_train=True,
                                                                    imgshape=imgshape_2,
                                                                    range_flow=range_flow, model_lvl1=model_lvl1).cuda()

    loss_similarity = MSE().loss
    loss_smooth = smoothloss
    # loss_Jdet = neg_Jdet_loss

    transform = SpatialTransform_unit().cuda()

    for param in transform.parameters():
        param.requires_grad = False
        param.volatile = True


    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    lossall = np.zeros((3, iteration_lvl2 + 1))

    step = 0
    while step <= iteration_lvl2:
        for i_iter in range(inner_iter):
        # for X, Y in training_generator:
            inputs, y_true = next(generator)

            inputs = [torch.from_numpy(d).to('cuda').float().permute(0, 3, 1, 2) for d in inputs]
            y_true = [torch.from_numpy(d).to('cuda').float().permute(0, 3, 1, 2) for d in y_true]

            X, Y = inputs[0], y_true[0]

            X = X.squeeze(-1).cuda().float()
            Y = Y.squeeze(-1).cuda().float()
            reg_code = torch.rand(1, dtype=X.dtype, device=X.device).unsqueeze(dim=0)

            F_X_Y, X_Y, Y_4x, F_xy, F_xy_lvl1, _ = model(X, Y, reg_code)

            loss_multiNCC = loss_similarity(X_Y, Y_4x)

            # F_X_Y_norm = transform_unit_flow_to_flow_cuda(F_X_Y.permute(0, 2, 3, 1).clone())

            # loss_Jacobian = loss_Jdet(F_X_Y_norm, grid_2)

            _, _, x, y = F_X_Y.shape
            norm_vector = torch.zeros((1, 2, 1, 1), dtype=F_X_Y.dtype, device=F_X_Y.device)
            norm_vector[0, 0, 0, 0] = (y - 1)
            norm_vector[0, 1, 0, 0] = (x - 1)
            loss_regulation = loss_smooth(F_X_Y * norm_vector)

            smo_weight = reg_code * max_smooth
            loss = loss_multiNCC + smo_weight * loss_regulation

            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients

            lossall[:, step] = np.array(
                [loss.item(), loss_multiNCC.item(), loss_regulation.item()])
            sys.stdout.write(
                "\r" + 'step "{0}" -> training loss "{1:.4f}" - sim_NCC "{2:4f}" -smo "{3:.4f} -reg_c "{4:.4f}"'.format(
                    step, loss.item(), loss_multiNCC.item(), loss_regulation.item(),
                    reg_code[0].item()))
            sys.stdout.flush()

            # with lr 1e-3 + with bias
            if (step % n_checkpoint == 0):
                modelname = model_dir + "/stagelvl2_" + str(step) + '.pth'
                torch.save(model.state_dict(), modelname)
                np.save(model_dir + '/loss' + "/stagelvl2_" + str(step) + '.npy', lossall)

                #########################################
                # set up wandb tracking
                #########################################
                src_im = wandb.Image(inputs[0].cpu().detach().numpy()[0, 0])
                tgt_im = wandb.Image(inputs[1].cpu().detach().numpy()[0, 0])
                tgt_im_curr_lv = wandb.Image(Y_4x.cpu().detach().numpy()[0, 0])
                pred_tgt_curr_lv = wandb.Image(X_Y.cpu().detach().numpy()[0, 0])

                logs = {
                    "loss": loss.item(),
                    "sim_NCC": loss_multiNCC.item(),
                    "smo": loss_regulation.item(),
                    "reg_c": reg_code[0].item(),
                    "src_im": src_im,
                    "tgt_im": tgt_im,
                    "tgt_im_curr_lv": tgt_im_curr_lv,
                    "pred_tgt_curr_lv": pred_tgt_curr_lv,
                }

                wandb.log(logs)

            if step == freeze_step:
                model.unfreeze_modellvl1()

            step += 1

            if step > iteration_lvl2:
                break
        print("one epoch pass")
    np.save(model_dir + '/loss' + '/stagelvl2.npy', lossall)


def train_lvl3():
    print("Training lvl3...")
    model_lvl1 = Miccai2021_LDR_conditional_laplacian_unit_disp_add_lvl1(2, 2, start_channel, is_train=True,
                                                                         imgshape=imgshape_4,
                                                                         range_flow=range_flow).cuda()
    model_lvl2 = Miccai2021_LDR_conditional_laplacian_unit_disp_add_lvl2(2, 2, start_channel, is_train=True,
                                                                         imgshape=imgshape_2,
                                                                         range_flow=range_flow,
                                                                         model_lvl1=model_lvl1).cuda()

    model_path = sorted(glob.glob(model_dir + "/stagelvl2_?????.pth"))[-1]
    # model_path = sorted(glob.glob(model_dir + "/stagelvl2_?.pth"))[-1]

    model_lvl2.load_state_dict(torch.load(model_path))
    print("Loading weight for model_lvl2...", model_path)

    # Freeze model_lvl1 weight
    for param in model_lvl2.parameters():
        param.requires_grad = False

    model = Miccai2021_LDR_conditional_laplacian_unit_disp_add_lvl3(2, 2, start_channel, is_train=True,
                                                                    imgshape=imgshape,
                                                                    range_flow=range_flow, model_lvl2=model_lvl2).cuda()

    loss_similarity = MSE().loss
    loss_smooth = smoothloss

    transform = SpatialTransform_unit().cuda()
    transform_nearest = SpatialTransformNearest_unit().cuda()

    for param in transform.parameters():
        param.requires_grad = False
        param.volatile = True



    grid_unit = F.affine_grid(torch.eye(3)[0:2].unsqueeze(0), (1,) + (1,) + imgshape, align_corners=True).cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    

    lossall = np.zeros((3, iteration_lvl3 + 1))

   
    step = 0
    while step <= iteration_lvl3:
        for i_iter in range(inner_iter):
            inputs, y_true = next(generator)

            inputs = [torch.from_numpy(d).to('cuda').float().permute(0, 3, 1, 2) for d in inputs]
            y_true = [torch.from_numpy(d).to('cuda').float().permute(0, 3, 1, 2) for d in y_true]

            X, Y = inputs[0], y_true[0]

        # for X, Y in training_generator:

            X = X.squeeze(-1).cuda().float()
            Y = Y.squeeze(-1).cuda().float()
            reg_code = torch.rand(1, dtype=X.dtype, device=X.device).unsqueeze(dim=0)

            F_X_Y, X_Y, Y_4x, F_xy, F_xy_lvl1, F_xy_lvl2, _ = model(X, Y, reg_code)

            loss_multiNCC = loss_similarity(X_Y, Y_4x)

            _, _, x, y = F_X_Y.shape
            norm_vector = torch.zeros((1, 2, 1, 1), dtype=F_X_Y.dtype, device=F_X_Y.device)
            norm_vector[0, 0, 0, 0] = (y - 1)
            norm_vector[0, 1, 0, 0] = (x - 1)
            loss_regulation = loss_smooth(F_X_Y * norm_vector)

            smo_weight = reg_code * max_smooth
            loss = loss_multiNCC + smo_weight * loss_regulation

            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients

            lossall[:, step] = np.array(
                [loss.item(), loss_multiNCC.item(), loss_regulation.item()])
            sys.stdout.write(
                "\r" + 'step "{0}" -> training loss "{1:.4f}" - sim_NCC "{2:4f}" -smo "{3:.4f} -reg_c "{4:.4f}"'.format(
                    step, loss.item(), loss_multiNCC.item(), loss_regulation.item(),
                    reg_code[0].item()))
            sys.stdout.flush()

            # with lr 1e-3 + with bias
            if (step % n_checkpoint == 0):
                modelname = model_dir  + "/stagelvl3_" + str(step) + '.pth'
                torch.save(model.state_dict(), modelname)
                np.save(model_dir + '/loss' + "/stagelvl3_" + str(step) + '.npy', lossall)

                #########################################
                # set up wandb tracking
                #########################################
                src_im = wandb.Image(inputs[0].cpu().detach().numpy()[0, 0])
                tgt_im = wandb.Image(inputs[1].cpu().detach().numpy()[0, 0])
                tgt_im_curr_lv = wandb.Image(Y_4x.cpu().detach().numpy()[0, 0])
                pred_tgt_curr_lv = wandb.Image(X_Y.cpu().detach().numpy()[0, 0])

                logs = {
                    "loss": loss.item(),
                    "sim_NCC": loss_multiNCC.item(),
                    "smo": loss_regulation.item(),
                    "reg_c": reg_code[0].item(),
                    "src_im": src_im,
                    "tgt_im": tgt_im,
                    "tgt_im_curr_lv": tgt_im_curr_lv,
                    "pred_tgt_curr_lv": pred_tgt_curr_lv,
                }

                wandb.log(logs)

            if step == freeze_step:
                model.unfreeze_modellvl2()

            step += 1

            if step > iteration_lvl3:
                break
        print("one epoch pass") 
    np.save(model_dir + '/loss' + '/stagelvl3.npy', lossall)


if __name__ == "__main__":
    imgshape = (128, 128)
    imgshape_4 = (128 // 4, 128 // 4)
    imgshape_2 = (128 // 2, 128 // 2)


    # range_flow = 1 # 0.4
    # max_smooth = 0.1 # 10
    start_t = datetime.now()
    train_lvl1()
    train_lvl2()
    train_lvl3()
    # time
    end_t = datetime.now()
    total_t = end_t - start_t
    print("Time: ", total_t.total_seconds())
