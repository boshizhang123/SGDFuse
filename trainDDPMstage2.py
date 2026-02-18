import torch
import models as Model
import argparse
import logging
import core.logger as Logger
import core.metrics as Metrics
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import os
import numpy as np
from data.VIDataset import FusionDataset as FD

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/sgdfuse_stage2.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'val'],
                        help='Run either train(training) or val(generation)', default='train')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-log_eval', action='store_true')

    args = parser.parse_args()
    opt = Logger.parse(args)
    opt = Logger.dict_to_nonedict(opt)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    Logger.setup_logger(None, opt['path']['log'], 'train', level=logging.INFO, screen=True)
    Logger.setup_logger('val', opt['path']['log'], 'val', level=logging.INFO)
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))
    tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])

    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train' and args.phase != 'val':
            logger.info("Creating Stage II train dataloader.")
            train_dataset = FD(split='train',
                               crop_size=dataset_opt['resolution'],
                               is_crop=True) 
            logger.info("Training dataset length: {}".format(train_dataset.length))
            train_loader = DataLoader(
                dataset=train_dataset,
                batch_size=dataset_opt['batch_size'], 
                shuffle=True,
                num_workers=4,
                pin_memory=True,
                drop_last=True,
            )
            train_loader.n_iter = len(train_loader)

    logger.info('Initial Dataset Finished')

    diffusion = Model.create_model(opt)
    logger.info('Initial SGDFuse Stage II Model Finished')

    current_step = diffusion.begin_step
    current_epoch = diffusion.begin_epoch
    n_iter = opt['train']['n_iter'] 

    if opt['path']['resume_state']:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(current_epoch, current_step))

    diffusion.set_new_noise_schedule(opt['model']['beta_schedule'][opt['phase']], schedule_phase=opt['phase'])

    if opt['phase'] == 'train':
        while current_step < n_iter:
            current_epoch += 1
            for _, (train_data, _) in enumerate(train_loader):
                current_step += 1
                if current_step > n_iter:
                    break
                
                diffusion.feed_data(train_data)
                diffusion.optimize_parameters()

                if current_step % opt['train']['print_freq'] == 0:
                    logs = diffusion.get_current_log()
                    message = '<epoch:{:3d}, iter:{:8,d}> '.format(current_epoch, current_step)
                    for k, v in logs.items():
                        message += '{:s}: {:.4e} '.format(k, v)
                        tb_logger.add_scalar(k, v, current_step)
                    logger.info(message)

                if current_step % opt['train']['val_freq'] == 0:
                    result_path = '{}/{}'.format(opt['path']['results'], current_epoch)
                    os.makedirs(result_path, exist_ok=True)

                    diffusion.set_new_noise_schedule(opt['model']['beta_schedule']['val'], schedule_phase='val')
                    
                    for idx in range(0, min(opt['datasets']['val']['data_len'], 4)):
                        diffusion.test_concat(in_channels=5, 
                                              img_size_w=opt['datasets']['val']['image_size_w'],
                                              img_size_h=opt['datasets']['val']['image_size_h'],
                                              x_f1=train_data['f1'][idx:idx+1, ...],
                                              x_sam_ir=train_data['sam_ir'][idx:idx+1, ...],
                                              x_sam_vis=train_data['sam_vis'][idx:idx+1, ...],
                                              continous=False)

                        visuals = diffusion.get_current_visuals()
                        final_fusion = Metrics.tensor2img(visuals['SAM']) 

                        Metrics.save_img(final_fusion, '{}/sample_iter{}_idx{}.png'.format(result_path, current_step, idx))
                        tb_logger.add_image('Val_Result_Idx_{}'.format(idx), 
                                            np.transpose(final_fusion, [2, 0, 1]), current_step)

                    diffusion.set_new_noise_schedule(opt['model']['beta_schedule']['train'], schedule_phase='train')
                    logger_val = logging.getLogger('val')
                    logger_val.info('<epoch:{:3d}, iter:{:8,d}> Semantic-guided generation sample saved.'.format(current_epoch, current_step))

                if current_step % opt['train']['save_checkpoint_freq'] == 0:
                    logger.info('Saving SGDFuse checkpoints at iteration {}.'.format(current_step))
                    diffusion.save_network(current_epoch, current_step)

        logger.info('End of SGDFuse Stage II training.')
