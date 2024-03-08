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
    smoothloss, \
    neg_Jdet_loss, WeightedMSE

import data_generators as data_gen

import model.networks as networks
import model.losses_covis as covis_losses_bank

parser = ArgumentParser()
parser.add_argument("--lr", type=float,
                    dest="lr", default=1e-4, help="learning rate")
# parser.add_argument("--iteration_lvl1", type=int,
#                     dest="iteration_lvl1", default=10001,
#                     help="number of lvl1 iterations")
# parser.add_argument("--iteration_lvl2", type=int,
#                     dest="iteration_lvl2", default=10001,
#                     help="number of lvl2 iterations")
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

parser.add_argument('--iter_covis_start', type=int, default=5000,
                    help='number of steps for covis to warm up')
parser.add_argument('--alpha', type=float, dest='alpha', default=0.01,
                    help='weight of scoring regularization loss (default: 0.01)')
parser.add_argument('--beta', type=float, dest='beta', default=0.1,
                    help='weight of variance regularization loss (default: 0.1)')

#########################

opt = parser.parse_args()

lr = opt.lr
start_channel = opt.start_channel
antifold = opt.antifold
n_checkpoint = opt.checkpoint
# datapath = opt.datapath
freeze_step = opt.freeze_step

# iteration_lvl1 = opt.iteration_lvl1
# iteration_lvl2 = opt.iteration_lvl2
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
        project="cvpr2024_covis_tv",
        
        # track hyperparameters and run metadata
        config={
        "learning_rate": opt.lr,
        "architecture": "cLapIRN",
        "dataset": opt.dataset,
        # "iteration_lvl1": opt.iteration_lvl1,
        # "iteration_lvl2": opt.iteration_lvl2,
        "iteration_lvl3": opt.iteration_lvl3,
        "batch_size": opt.batch_size,
        "range_flow": opt.range_flow,
        "max_smooth": opt.max_smooth,
        "alpha": opt.alpha,
        "beta": opt.beta,
        },

        # have a run name
        name = opt.wandb_name
    )

model_dir = opt.model_dir

if not os.path.exists(model_dir):
    os.makedirs(model_dir)
    
if not os.path.exists(model_dir + '/loss'):
    os.makedirs(model_dir + '/loss')



def train_lvl3():
    print("Training lvl3...")
    model_lvl1 = Miccai2021_LDR_conditional_laplacian_unit_disp_add_lvl1(2, 2, start_channel, is_train=True,
                                                                         imgshape=imgshape_4,
                                                                         range_flow=range_flow).cuda()
    model_lvl2 = Miccai2021_LDR_conditional_laplacian_unit_disp_add_lvl2(2, 2, start_channel, is_train=True,
                                                                         imgshape=imgshape_2,
                                                                         range_flow=range_flow,
                                                                         model_lvl1=model_lvl1).cuda()

    print(os.listdir('./'))
    # print(os.listdir('./pretrained_model/'))
    # print(os.listdir('./pretrained_model/{}/'.format(str(opt.dataset))))
    model_path = './Conditional_LapIRN/pretrained_model/{}/{}_stagelvl2_10000.pth'.format(str(opt.dataset),str(opt.dataset))
    # model_path = sorted(glob.glob(model_dir + "/stagelvl2_?.pth"))[-1]

    model_lvl2.load_state_dict(torch.load(model_path))
    print("Loading weight for model_lvl2...", model_path)

    # Freeze model_lvl1 weight
    for param in model_lvl2.parameters():
        param.requires_grad = False

    model = Miccai2021_LDR_conditional_laplacian_unit_disp_add_lvl3(2, 2, start_channel, is_train=True,
                                                                    imgshape=imgshape,
                                                                    range_flow=range_flow, model_lvl2=model_lvl2).cuda()

    loss_similarity = WeightedMSE().loss
    loss_smooth = smoothloss


    #### covis loss ####
    loss_func_covis = covis_losses_bank.CoVisWeightedMSE().loss
    losses_covis = [loss_func_covis]
    weights_covis = [1]

    losses_covis += [covis_losses_bank.ScoringLoss().loss]
    weights_covis += [opt.alpha]

    losses_covis += [covis_losses_bank.CoVisTVLoss(ndims = 2).loss]
    weights_covis += [opt.beta]


    #### load covis model ####
    enc_nf = [16, 32, 32, 32]
    dec_nf = [32, 32, 32, 32, 32, 16, 16]
    # if args.load_model_covis:
    #     # load initial model (if specified)
    #     model_covis = networks.CoVisNet.load(args.load_model_covis, device)
    # else:
        # otherwise configure new model
    model_covis = networks.CoVisNet(
            inshape=imgshape,
            nb_unet_features=[enc_nf, dec_nf],
            bidir=False,
            int_steps=7,
            int_downsize=2
        )
    
    model_covis.to('cuda')
    model_covis.train()

    optimizer_covis = torch.optim.Adam(model_covis.parameters(), lr=1e-4)


    # transform = SpatialTransform_unit().cuda()
    # transform_nearest = SpatialTransformNearest_unit().cuda()

    # for param in transform.parameters():
    #     param.requires_grad = False
    #     param.volatile = True



    # grid_unit = F.affine_grid(torch.eye(3)[0:2].unsqueeze(0), (1,) + (1,) + imgshape, align_corners=True).cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    

    lossall = np.zeros((3, iteration_lvl3 + 1))

   
    step = 0
    while step <= iteration_lvl3:
        for i_iter in range(inner_iter):
            inputs, y_true = next(generator)

            inputs = [torch.from_numpy(d).to('cuda').float().permute(0, 3, 1, 2) for d in inputs]
            y_true = [torch.from_numpy(d).to('cuda').float().permute(0, 3, 1, 2) for d in y_true]

            X, Y = inputs[0], y_true[0]

            if step <= opt.iter_covis_start:
                scoring_mask = torch.ones_like(Y)
            else:
                scoring_mask = model_covis(Y)

        # for X, Y in training_generator:
            #########################################################
            # run the forward pass of motion estimator from t to t+1
            #########################################################

            X = X.squeeze(-1).cuda().float()
            Y = Y.squeeze(-1).cuda().float()
            reg_code = torch.rand(1, dtype=X.dtype, device=X.device).unsqueeze(dim=0)

            F_X_Y, X_Y, Y_4x, F_xy, F_xy_lvl1, F_xy_lvl2, _ = model(X, Y, reg_code)

            loss_multiNCC = loss_similarity(X_Y, Y_4x, scoring_mask)

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

            #########################################################
            # run the forward pass of covis estimator from t to t+1
            #########################################################
            F_X_Y, X_Y, Y_4x, F_xy, F_xy_lvl1, F_xy_lvl2, _ = model(X, Y, reg_code)

            covis_loss = 0
            covis_loss_list = []


            # for idx_bidir in range(num_directions):
            scoring_mask = model_covis(Y)
            # covis_vis.append(scoring_mask)


            for n, loss_function in enumerate(losses_covis):

                # for idx_bidir in range(num_directions):
                curr_loss = loss_function(y_true[0], X_Y, scoring_mask) * weights_covis[n]
                        
                covis_loss_list.append(curr_loss.item())
                covis_loss += curr_loss
                    
            # backpropagate and optimize t -> t+1
            optimizer_covis.zero_grad()
            covis_loss.backward()
            optimizer_covis.step()
                


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

                pred_scoring_mask = wandb.Image(scoring_mask.cpu().detach().numpy()[0, 0])

                logs = {
                    "loss": loss.item(),
                    "sim_NCC": loss_multiNCC.item(),
                    "smo": loss_regulation.item(),
                    "reg_c": reg_code[0].item(),
                    "src_im": src_im,
                    "tgt_im": tgt_im,
                    "tgt_im_curr_lv": tgt_im_curr_lv,
                    "pred_tgt_curr_lv": pred_tgt_curr_lv,

                    "loss_covis": covis_loss.item(),
                    "loss_covis_reg": covis_loss_list[1],
                    "loss_covis_tv": covis_loss_list[2],
                    "pred_scoring_mask": pred_scoring_mask,
                    "pred_scoring_mask_mean": np.mean(scoring_mask.cpu().detach().numpy()[0, 0]),
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
    # train_lvl1()
    # train_lvl2()
    train_lvl3()
    # time
    end_t = datetime.now()
    total_t = end_t - start_t
    print("Time: ", total_t.total_seconds())
