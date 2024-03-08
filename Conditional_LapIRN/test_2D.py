import os
from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn.functional as F
import glob

import data_generators as data_gen

# from Functions import save_img, save_flow, load_4D_with_header, imgnorm
from miccai2021_model_2D import Miccai2021_LDR_conditional_laplacian_unit_disp_add_lvl1, \
    Miccai2021_LDR_conditional_laplacian_unit_disp_add_lvl2, Miccai2021_LDR_conditional_laplacian_unit_disp_add_lvl3, \
    SpatialTransform_unit, SpatialTransformNearest_unit

from spatial_transform import SpatialTransformer

from scipy.io import loadmat, savemat

parser = ArgumentParser()

parser.add_argument('--dataset', required=True, help='Name of the dataset')
parser.add_argument('--test-dir', required=True, help='path to all test files')
parser.add_argument('--result-dir', required=True, help='where to save the result')
parser.add_argument('--model', required=True, help='pytorch model for nonlinear registration')


# parser.add_argument("--modelpath", type=str,
#                     dest="modelpath", default='../Model/LDR_OASIS_NCC_unit_disp_add_fea64_reg01_10_2D_stagelvl3_90000.pth',
#                     help="Trained model path")
# parser.add_argument("--savepath", type=str,
#                     dest="savepath", default='../Result',
#                     help="path for saving images")
parser.add_argument("--start_channel", type=int,
                    dest="start_channel", default=64,
                    help="number of start channels")
# parser.add_argument("--fixed", type=str,
#                     dest="fixed", default='../Data/image_A_2D.nii.gz',
#                     help="fixed image")
# parser.add_argument("--moving", type=str,
#                     dest="moving", default='../Data/image_B_2D.nii.gz',
#                     help="moving image")
parser.add_argument("--reg_input", type=float,
                    dest="reg_input", default=0.1,
                    help="Normalized smoothness regularization (within [0,1])")
opt = parser.parse_args()

test_files = glob.glob(os.path.join(opt.test_dir, '*.mat'))
assert len(test_files) > 0

inshape = (128, 128)

warp_layer = SpatialTransformer(inshape).to('cuda')

if not os.path.isdir(opt.result_dir):
    os.mkdir(opt.result_dir)

start_channel = opt.start_channel
reg_input = opt.reg_input


def test():
    print("Current reg_input: ", str(reg_input))

    model_lvl1 = Miccai2021_LDR_conditional_laplacian_unit_disp_add_lvl1(2, 2, start_channel, is_train=True, imgshape=imgshape_4,
                                                                         range_flow=range_flow).cuda()
    model_lvl2 = Miccai2021_LDR_conditional_laplacian_unit_disp_add_lvl2(2, 2, start_channel, is_train=True, imgshape=imgshape_2,
                                                                         range_flow=range_flow, model_lvl1=model_lvl1).cuda()

    model = Miccai2021_LDR_conditional_laplacian_unit_disp_add_lvl3(2, 2, start_channel, is_train=False, imgshape=imgshape,
                                                                    range_flow=range_flow, model_lvl2=model_lvl2).cuda()


    # transform = SpatialTransform_unit().cuda()
    # transform_nearest = SpatialTransformNearest_unit().cuda()

    transform = SpatialTransformer(imgshape).cuda()

    model.load_state_dict(torch.load(opt.model))
    model.eval()
    transform.eval()

    grid = F.affine_grid(torch.eye(3)[0:2].unsqueeze(0), (1,) + (1,) + imgshape, align_corners=True).cuda()

    use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")

    warp_layer = SpatialTransformer(inshape).to('cuda')



    for file_path in test_files:
        

        im_ED, im_ES, ED_myo, ES_myo = data_gen.py.utils_2D.load_imfile_mat(file_path, add_batch_axis=True, add_feat_axis=True, registration=True)

        input_ED = torch.from_numpy(im_ED).to(device).float().permute(0, 3, 1, 2)
        input_ES = torch.from_numpy(im_ES).to(device).float().permute(0, 3, 1, 2)
        input_ED_myo = torch.from_numpy(ED_myo).to(device).float().permute(0, 3, 1, 2)

        reg_code = torch.tensor([0.1], dtype=input_ES.dtype, device=input_ES.device).unsqueeze(dim=0)

        dvf = model(input_ED, input_ES, reg_code)

        ES_myo_pred = warp_layer(input_ED_myo, dvf.to(device))
        im_ES_pred = warp_layer(input_ED, dvf.to(device))

        save_file = {
                "im_ED": im_ED[0, :, :, 0], 
                "im_ES": im_ES[0, :, :, 0], 
                "im_ES_pred": im_ES_pred.detach().cpu().numpy()[0,0], 
                "ED_myo": ED_myo[0, :, :, 0], 
                "ES_myo":ES_myo[0, :, :, 0], 
                "ES_myo_pred": ES_myo_pred.detach().cpu().numpy()[0, 0], 
                "dvf": dvf.detach().cpu().numpy()[0],
                }

        save_name = file_path.split('/')[-1]
        savemat(os.path.join(opt.result_dir, save_name) ,save_file)





    # fixed_img, header, affine = load_4D_with_header(fixed_path)
    # moving_img, _, _ = load_4D_with_header(moving_path)

    # fixed_img, moving_img = imgnorm(fixed_img), imgnorm(moving_img)
    # fixed_img = torch.from_numpy(fixed_img).float().to(device).unsqueeze(dim=0).squeeze(-1)
    # moving_img = torch.from_numpy(moving_img).float().to(device).unsqueeze(dim=0).squeeze(-1)

    # with torch.no_grad():
    #     reg_code = torch.tensor([reg_input], dtype=fixed_img.dtype, device=fixed_img.device).unsqueeze(dim=0)

    #     F_X_Y = model(moving_img, fixed_img, reg_code)

    #     X_Y = transform(moving_img, F_X_Y.permute(0, 2, 3, 1), grid).unsqueeze(-1).data.cpu().numpy()[0, 0, :, :, :]

    #     F_X_Y_cpu = F_X_Y.unsqueeze(-1).data.cpu().numpy()[0, :, :, :, :].transpose(1, 2, 3, 0)
    #     x, y, z, _ = F_X_Y_cpu.shape
    #     F_X_Y_cpu[:, :, :, 0] = F_X_Y_cpu[:, :, :, 0] * (y - 1) / 2
    #     F_X_Y_cpu[:, :, :, 1] = F_X_Y_cpu[:, :, :, 1] * (x - 1) / 2

    #     save_flow(F_X_Y_cpu, savepath + '/warpped_flow_2D_' + 'reg' + str(reg_input) + '.nii.gz', header=header, affine=affine)
    #     save_img(X_Y, savepath + '/warpped_moving_2D_' + 'reg' + str(reg_input) + '.nii.gz', header=header, affine=affine)

    # print("Result saved to :", savepath)


if __name__ == '__main__':
    imgshape = (128, 128)
    imgshape_4 = (128 // 4, 128 // 4)
    imgshape_2 = (128 // 2, 128 // 2)

    range_flow = 1
    test()
