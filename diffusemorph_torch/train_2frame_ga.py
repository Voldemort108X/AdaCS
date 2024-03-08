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

    # loss
    parser.add_argument('--image-loss', required=True,
                    help='image reconstruction loss - can be mse or ncc (default: mse)')
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
    
    # gradient accumulation
    parser.add_argument('--accumulation_steps', type=int, default=1, help='number of steps before backward and optimizer step')

    parser.add_argument('-debug', '-d', action='store_true')

    # wandb run name
    parser.add_argument('--wandb-name', type=str, required=True, help='name of wandb run')

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

    # loss weights
    opt['model']['loss_lambda'] = args.lambda_L
    opt['model']['loss_gamma'] = args.gamma

    # get datashape
    inshape = next(generator)[0][0].shape[1:-1]
    ndims = len(inshape)

    if ndims == 3:
        opt['model']['diffusion']['image_size'] = inshape
    elif ndims == 2:
        assert inshape[0] == inshape[1]
        opt['model']['diffusion']['image_size'] = inshape[0]
    opt['model']['motion_loss_type'] = args.image_loss

    # for phase, dataset_opt in opt['datasets'].items():
    #     if phase != opt['phase']: continue
    #     if opt['phase'] == 'train':
    #         batchSize = opt['datasets']['train']['batch_size']
    #         # train_set = Data.create_dataset_3D(dataset_opt, phase)
    #         # train_loader = Data.create_dataloader(train_set, dataset_opt, phase)
    #         training_iters = args.steps_per_epoch
    #     # elif opt['phase'] == 'test':
    #     #     test_set = Data.create_dataset_3D(dataset_opt, phase)
    #     #     test_loader = Data.create_dataloader(test_set, dataset_opt, phase)
    logger.info('Initial Dataset Finished')


    # wandb tracking
    # wandb tracking experiments
    run = wandb.init(
        # set the wandb project where this run will be logged
        project="cardiac_motion_baselines",
        
        # track hyperparameters and run metadata
        config={
        "learning_rate": opt['train']['optimizer']['lr'],
        "architecture": "DiffuseMorph",
        "dataset": args.dataset,
        "epochs": opt['train']['n_epoch'],
        "batch_size": args.batch_size,
        "steps_per_epoch": args.steps_per_epoch,
        "image-loss": opt['model']['motion_loss_type'],
        "loss_weights": opt['model']['loss_lambda'],
        "loss_gamma": opt['model']['loss_gamma'],
        "accumulation_steps": args.accumulation_steps
        },

        # have a run name
        name = args.wandb_name
    )


    # model
    diffusion = Model.create_model(opt)
    logger.info('Initial Model Finished')

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

                # convert to DiffuseMorph format
                train_data = {'M': inputs[0], 'F': y_true[0]}

                current_step += 1

                diffusion.feed_data_2frame(train_data)

                # print(current_step)

                # diffusion.optimize_parameters_2frame()

                diffusion.optimize_parameters_2frame_grad_accum(idx_step=istep, grad_accum_step=args.accumulation_steps)


                # log
                # if (istep + 1) % opt['train']['print_freq'] == 0:
                #     logs = diffusion.get_current_log()
                #     t = (time.time() - iter_start_time) / args.batch_size
                # logs, img_logs = diffusion.get_current_log()
                # logs content:  OrderedDict([('l_pix', xxx), ('l_sim', xxx), ('l_smt', xxx), ('l_tot', xxx)])
                # img_logs content: ['x_recon', 'out_M']
                # print(img_logs.keys()) # x_recon, out_M
                # print(img_logs['x_recon'].shape) # 1 x 1 x 64 x 64 x 64
                # print(img_logs['out_M'].shape) # 1 x 1 x 64 x 64 x 64

            # save for each epoch
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
                
            elif ndims == 2:
                src_im = wandb.Image(inputs[0].cpu().detach().numpy()[0, 0])
                tgt_im = wandb.Image(inputs[1].cpu().detach().numpy()[0, 0])
                pred_tgt = wandb.Image(img_logs['out_M'].cpu().detach().numpy()[0, 0])
                x_recon = wandb.Image(img_logs['x_recon'].cpu().detach().numpy()[0, 0])
                mask_bgd = wandb.Image(img_logs['mask_bgd'].cpu().detach().numpy()[0, 0])

            logs['src_im'] =  src_im
            logs['tgt_im'] = tgt_im
            logs['pred_tgt'] = pred_tgt
            logs['x_recon'] = x_recon
            logs['mask_bgd'] = mask_bgd

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

        # save model

        logger.info('End of training.')
