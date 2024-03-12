import torch
import numpy as np

class DimensionConverterDP():
    '''
    going to and from [0, dim_size] and [-1, 1]
    '''
    def __init__(self, shape):
        self.shape = np.array(shape) - 1 # minus 1 here b/c if img has size 64, we want [0,63] range
    
    def to_orig_img_dim(self, x):
        m = torch.tensor(self.shape/2, dtype=torch.get_default_dtype(), device=x.device)
        b = torch.tensor(self.shape/2, dtype=torch.get_default_dtype(), device=x.device)
        
        y = m*x + b
        
        return y
    
    def from_orig_img_dim(self, x):
        m = torch.tensor(2/self.shape, dtype=torch.get_default_dtype(), device=x.device)
        b = -1
        
        y = m*x + b
        
        return y

def apply_linear_transform_on_img_torch(src_data, src_to_tgt_transformation, tgt_shape, grid_sample_mode='bilinear'):
    '''
    decide which source voxels to sample, starting from target voxel coordinates
    src_data = img_cuda[None,None,:,:,:]
    src_to_tgt_transformation: homogeneous coordinate frame in 3D space, shape (4,4)
    tgt_shape = [128,128,128]
    '''
    x, y, z = torch.arange(tgt_shape[0]), torch.arange(tgt_shape[1]), torch.arange(tgt_shape[2])
    xmesh, ymesh, zmesh = torch.meshgrid(x, y, z) # (x.shape, y.shape, z.shape)
    coords_stack = torch.stack([xmesh, ymesh, zmesh], dim=0).reshape(3,-1) # (3, np.prod(tgt_shape))
    homo_coords = torch.cat([coords_stack, torch.ones([1,coords_stack.shape[1]])], dim=0).to(dtype=torch.get_default_dtype(), device=src_data.device) # (4, np.prod(tgt_shape))

    dim_conv = DimensionConverterDP(src_data.shape[-3:])
    # print(src_data.shape[-3:])
    with torch.no_grad():
        sample_coords = torch.matmul(torch.linalg.inv(src_to_tgt_transformation), homo_coords)[:3].T # (np.prod(tgt_shape), 3)
        grid = dim_conv.from_orig_img_dim(sample_coords).reshape(*tgt_shape,-1)[None,:,:,:,:].flip([-1]) # (1, *tgt_shape, 3)
        new_img = torch.nn.functional.grid_sample(src_data, grid, align_corners=True, mode=grid_sample_mode) # (1, 1, *tgt_shape)
    return new_img


def imresize3D(img, target_shape):
    # assume img: HxWxD
    diag_arr = np.append(np.array(target_shape) / np.array(img.shape), 1)
    src_to_tgt_transformation = np.diag(diag_arr)
    img_resize = apply_linear_transform_on_img_torch(torch.tensor(img, dtype=torch.get_default_dtype())[None,None,:,:,:], torch.tensor(src_to_tgt_transformation, dtype=torch.get_default_dtype()), target_shape)[0,0].numpy()
    
    return img_resize


def delete_zeros(im, mask_bgd):
    im = im.flatten()
    mask_bgd = mask_bgd.flatten()
    im = im[mask_bgd!=0]
    
    return im


def dvf_resize(dvf, tgt_shape):
    # assume dvf in shape (3, x, y, z)
    assert dvf.shape[0] == 3

    dvf_x_resize = imresize3D(dvf[0], tgt_shape) * tgt_shape[0] / dvf.shape[1]
    dvf_y_resize = imresize3D(dvf[1], tgt_shape) * tgt_shape[1] / dvf.shape[2]
    dvf_z_resize = imresize3D(dvf[2], tgt_shape) * tgt_shape[2] / dvf.shape[3]
    dvf_resize = np.stack((dvf_x_resize, dvf_y_resize, dvf_z_resize), axis=0)

    return dvf_resize



def load_test_echo_noresize(file_noresize):
    # generate ED_myo
    if 'ED_idx' in file_noresize.keys():
        myo_ED = file_noresize['epi_ED_resize'] - file_noresize['endo_ED_resize']
        myo_ES = file_noresize['epi_ES_resize'] - file_noresize['endo_ES_resize']

        im_ED = file_noresize['im4D_resize'][int(file_noresize['ED_idx'].squeeze()), :, :, :]
        im_ES = file_noresize['im4D_resize'][int(file_noresize['ES_idx'].squeeze()), :, :, :]
    else:
        numOfFrames = file_noresize['im4D_resize'].shape[-1]
        ED_idx = 0
        ES_idx = numOfFrames // 2

        myo_ED = file_noresize['epi4D_resize'][:, :, :, ED_idx] - file_noresize['endo4D_resize'][:, :, :, ED_idx]
        myo_ES = file_noresize['epi4D_resize'][:, :, :, ES_idx] - file_noresize['endo4D_resize'][:, :, :, ES_idx]

        im_ED = file_noresize['im4D_resize'][:, :, :, ED_idx]
        im_ES = file_noresize['im4D_resize'][:, :, :, ES_idx]
    
    return im_ED, im_ES, myo_ED, myo_ES



def compute_tv(input):
    # input: (x, y, z)
    dx = np.abs(input[1:, :, :] - input[:-1, :, :])
    dy = np.abs(input[:, 1:, :] - input[:, :-1, :])
    dz = np.abs(input[:, :, 1:] - input[:, :, :-1])

    d = np.mean(dx ** 2) + np.mean(dy ** 2) + np.mean(dz ** 2)

    return d / 3.0

def compute_tv_2d(input):
    # input: (x, y)
    dx = np.abs(input[1:, :] - input[:-1, :])
    dy = np.abs(input[:, 1:] - input[:, :-1])

    d = np.mean(dx ** 2) + np.mean(dy ** 2)

    return d / 2.0
