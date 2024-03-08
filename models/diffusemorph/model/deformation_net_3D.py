import torch
import torch.nn as nn
import torch.nn.functional as nnf

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


##############################################################################
# Networks
##############################################################################
class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        self.size = size

        # print('size', size)

        # print('size', size)
        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)

    def forward(self, src, flow):
        # print('self size', self.size)
        # print('src shape', src.shape)
        # print('flow shape', flow.shape)
        # print('flow shape', flow.shape[2:])
        # print('grid shape', self.grid.shape)
        # new locations
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return nnf.grid_sample(src, new_locs, align_corners=True, mode=self.mode)



# Spatial Transformer 3D Net #################################################
# class Dense3DSpatialTransformer(nn.Module):
#     def __init__(self):
#         super(Dense3DSpatialTransformer, self).__init__()

#     def forward(self, input1, input2):
#         return self._transform(input1, input2[:, 0], input2[:, 1], input2[:, 2])

#     def _transform(self, input1, dDepth, dHeight, dWidth):
#         batchSize = dDepth.shape[0]
#         dpt = dDepth.shape[1]
#         hgt = dDepth.shape[2]
#         wdt = dDepth.shape[3]

#         D_mesh, H_mesh, W_mesh = self._meshgrid(dpt, hgt, wdt)
#         D_mesh = D_mesh.unsqueeze_(0).expand(batchSize, dpt, hgt, wdt)
#         H_mesh = H_mesh.unsqueeze_(0).expand(batchSize, dpt, hgt, wdt)
#         W_mesh = W_mesh.unsqueeze_(0).expand(batchSize, dpt, hgt, wdt)
#         D_upmesh = dDepth + D_mesh
#         H_upmesh = dHeight + H_mesh
#         W_upmesh = dWidth + W_mesh

#         return self._interpolate(input1, D_upmesh, H_upmesh, W_upmesh)

#     def _meshgrid(self, dpt, hgt, wdt):
#         d_t = torch.linspace(0.0, dpt-1.0, dpt).unsqueeze_(1).unsqueeze_(1).expand(dpt, hgt, wdt).cuda()
#         h_t = torch.matmul(torch.linspace(0.0, hgt-1.0, hgt).unsqueeze_(1), torch.ones((1,wdt))).cuda()
#         h_t = h_t.unsqueeze_(0).expand(dpt, hgt, wdt)
#         w_t = torch.matmul(torch.ones((hgt,1)), torch.linspace(0.0, wdt-1.0, wdt).unsqueeze_(1).transpose(1,0)).cuda()
#         w_t = w_t.unsqueeze_(0).expand(dpt, hgt, wdt)
#         return d_t, h_t, w_t

#     def _interpolate(self, input, D_upmesh, H_upmesh, W_upmesh):
#         nbatch = input.shape[0]
#         nch    = input.shape[1]
#         depth  = input.shape[2]
#         height = input.shape[3]
#         width  = input.shape[4]

#         img = torch.zeros(nbatch, nch, depth+2,  height+2, width+2).cuda()
#         img[:, :, 1:-1, 1:-1, 1:-1] = input

#         imgDpt = img.shape[2]
#         imgHgt = img.shape[3]  # 256+2 = 258
#         imgWdt = img.shape[4]  # 256+2 = 258

#         # D_upmesh, H_upmesh, W_upmesh = [D, H, W] -> [BDHW,]
#         D_upmesh = D_upmesh.view(-1).float()+1.0  # (BDHW,)
#         H_upmesh = H_upmesh.view(-1).float()+1.0  # (BDHW,)
#         W_upmesh = W_upmesh.view(-1).float()+1.0  # (BDHW,)

#         # D_upmesh, H_upmesh, W_upmesh -> Clamping into [0, 257] -- index
#         df = torch.floor(D_upmesh).int()
#         dc = df + 1
#         hf = torch.floor(H_upmesh).int()
#         hc = hf + 1
#         wf = torch.floor(W_upmesh).int()
#         wc = wf + 1

#         df = torch.clamp(df, 0, imgDpt-1)  # (BDHW,)
#         dc = torch.clamp(dc, 0, imgDpt-1)  # (BDHW,)
#         hf = torch.clamp(hf, 0, imgHgt-1)  # (BDHW,)
#         hc = torch.clamp(hc, 0, imgHgt-1)  # (BDHW,)
#         wf = torch.clamp(wf, 0, imgWdt-1)  # (BDHW,)
#         wc = torch.clamp(wc, 0, imgWdt-1)  # (BDHW,)

#         # Find batch indexes
#         rep = torch.ones([depth*height*width, ]).unsqueeze_(1).transpose(1, 0).cuda()
#         bDHW = torch.matmul((torch.arange(0, nbatch).float()*imgDpt*imgHgt*imgWdt).unsqueeze_(1).cuda(), rep).view(-1).int()

#         # Box updated indexes
#         HW = imgHgt*imgWdt
#         W = imgWdt
#         # x: W, y: H, z: D
#         idx_000 = bDHW + df*HW + hf*W + wf
#         idx_100 = bDHW + dc*HW + hf*W + wf
#         idx_010 = bDHW + df*HW + hc*W + wf
#         idx_110 = bDHW + dc*HW + hc*W + wf
#         idx_001 = bDHW + df*HW + hf*W + wc
#         idx_101 = bDHW + dc*HW + hf*W + wc
#         idx_011 = bDHW + df*HW + hc*W + wc
#         idx_111 = bDHW + dc*HW + hc*W + wc

#         # Box values
#         img_flat = img.view(-1, nch).float()  # (BDHW,C) //// C=1

#         val_000 = torch.index_select(img_flat, 0, idx_000.long())
#         val_100 = torch.index_select(img_flat, 0, idx_100.long())
#         val_010 = torch.index_select(img_flat, 0, idx_010.long())
#         val_110 = torch.index_select(img_flat, 0, idx_110.long())
#         val_001 = torch.index_select(img_flat, 0, idx_001.long())
#         val_101 = torch.index_select(img_flat, 0, idx_101.long())
#         val_011 = torch.index_select(img_flat, 0, idx_011.long())
#         val_111 = torch.index_select(img_flat, 0, idx_111.long())

#         dDepth  = dc.float() - D_upmesh
#         dHeight = hc.float() - H_upmesh
#         dWidth  = wc.float() - W_upmesh

#         wgt_000 = (dWidth*dHeight*dDepth).unsqueeze_(1)
#         wgt_100 = (dWidth * dHeight * (1-dDepth)).unsqueeze_(1)
#         wgt_010 = (dWidth * (1-dHeight) * dDepth).unsqueeze_(1)
#         wgt_110 = (dWidth * (1-dHeight) * (1-dDepth)).unsqueeze_(1)
#         wgt_001 = ((1-dWidth) * dHeight * dDepth).unsqueeze_(1)
#         wgt_101 = ((1-dWidth) * dHeight * (1-dDepth)).unsqueeze_(1)
#         wgt_011 = ((1-dWidth) * (1-dHeight) * dDepth).unsqueeze_(1)
#         wgt_111 = ((1-dWidth) * (1-dHeight) * (1-dDepth)).unsqueeze_(1)

#         output = (val_000*wgt_000 + val_100*wgt_100 + val_010*wgt_010 + val_110*wgt_110 +
#                   val_001 * wgt_001 + val_101 * wgt_101 + val_011 * wgt_011 + val_111 * wgt_111)
#         output = output.view(nbatch, depth, height, width, nch).permute(0, 4, 1, 2, 3)  #B, C, D, H, W
#         return output


class Cblock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super(Cblock, self).__init__()
        self.block = nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=True)

    def forward(self, x):
        return self.block(x)

class CRblock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super(CRblock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=True),
            nn.LeakyReLU(0.2, True))

    def forward(self, x):
        return self.block(x)

class inblock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=2):
        super(inblock, self).__init__()
        self.block = CRblock(in_ch, out_ch, stride=stride)

    def forward(self, x):
        return self.block(x)

class outblock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=2, output_padding=1):
        super(outblock, self).__init__()
        self.block = nn.Conv3d(in_ch, out_ch, 3, padding=1, stride=stride)

    def forward(self, x):
        x = self.block(x)
        return x

class downblock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(downblock, self).__init__()
        self.block = CRblock(in_ch, out_ch, stride=2)

    def forward(self, x):
        return self.block(x)

class upblock(nn.Module):
    def __init__(self, in_ch, CR_ch, out_ch):
        super(upblock, self).__init__()
        self.upconv = nn.ConvTranspose3d(in_ch, in_ch, 3, padding=1, stride=2, output_padding=1)
        self.block = CRblock(CR_ch, out_ch)

    def forward(self, x1, x2):
        upconved = self.upconv(x1)
        x = torch.cat([x2, upconved], dim=1)
        return self.block(x)
#
class registUnetBlock(nn.Module):
    def __init__(self, input_nc, encoder_nc, decoder_nc, size):
        super(registUnetBlock, self).__init__()
        self.inconv = inblock(input_nc, encoder_nc[0], stride=1)
        self.downconv1 = downblock(encoder_nc[0], encoder_nc[1])
        self.downconv2 = downblock(encoder_nc[1], encoder_nc[2])
        self.downconv3 = downblock(encoder_nc[2], encoder_nc[3])
        self.downconv4 = downblock(encoder_nc[3], encoder_nc[4])
        self.upconv1 = upblock(encoder_nc[4], encoder_nc[4]+encoder_nc[3], decoder_nc[0])
        self.upconv2 = upblock(decoder_nc[0], decoder_nc[0]+encoder_nc[2], decoder_nc[1])
        self.upconv3 = upblock(decoder_nc[1], decoder_nc[1]+encoder_nc[1], decoder_nc[2])
        self.keepblock = CRblock(decoder_nc[2], decoder_nc[3])
        self.upconv4 = upblock(decoder_nc[3], decoder_nc[3]+encoder_nc[0], decoder_nc[4])
        self.outconv = outblock(decoder_nc[4], decoder_nc[5], stride=1)
        self.stn = SpatialTransformer(size)

    def forward(self, input):
        x1 = self.inconv(input)
        x2 = self.downconv1(x1)
        x3 = self.downconv2(x2)
        x4 = self.downconv3(x3)
        x5 = self.downconv4(x4)
        x = self.upconv1(x5, x4)
        x = self.upconv2(x, x3)
        x = self.upconv3(x, x2)
        x = self.keepblock(x)
        x = self.upconv4(x, x1)
        flow = self.outconv(x)
        # mov = (input[:, :1] + 1) / 2.0
        mov = input[:, :1]
        out = self.stn(mov, flow)
        return out, flow