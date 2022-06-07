import glob
import socket
from loss_function.triplet_loss import TripletLoss
from loss_function.ap_loss import AveragePrecisionLoss
from model.C3D_Hash_Model import C3D
from model.Res3D_Hash_Model import Res3D_Hash_Model
from parse_config import parse_opts
import torch
import os
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from utils import video2frames
from utils import load_data
from data_loader.data_loader import CustomDataset
import time
from utils import split_dataset
import torch.nn.functional as F
from model.Classify_Model import Classify
from tqdm import tqdm
import random
import pandas as pd
from datetime import datetime
from utils_log import compute_mAP
import matplotlib.pyplot as plt
import numpy as np

from utils_log.perf_log import logger2

def get_opt():
    opt = parse_opts()
    return opt

def main_worker(opt):
    logger2.info('Come into the function of main_worker')

    logger2.info('  Start pre-processing the video: ')
    dir_video_data = os.path.join('data/', opt.video_dataset)
    logger2.info('      The path of video data : ' + str(dir_video_data))
    dir_video_frames = 'data_frames/' + str(opt.sample_frames_nums) + '-' + str(opt.video_dataset)
    logger2.info('      The path of video frames : ' + str(dir_video_frames))
    path_video_label_pairs_file = video2frames(dir_video_data, dir_video_frames, opt.sample_frames_nums)
    logger2.info('      Path of video and label pairs file : ' + str(path_video_label_pairs_file))
    """ no need  """
    logger2.info('      Check out all video frames numbers : ')
    with open(path_video_label_pairs_file, 'r') as file:
        list_01 = file.readlines()
        file.close()
    for path_label in list_01:
        path = str(path_label).split(' ')[0]
        assert opt.sample_frames_nums == len(os.listdir(path)), path
    logger2.info('      All video frame numbers meet the requirements!')
    logger2.info('  Finish all the pre-processing of the videos! ')

    # split data set
    logger2.info('  Start split the video dataset to train(0.6), test(0.2), val(0.2): ')
    train_file, test_file, val_file = split_dataset(path_video_label_pairs_file, 0.6, 0.2, 0.2)
    logger2.info('      train_file:' + str(train_file) + '  test_file:' + str(test_file) + '  val_file:' + str(val_file))
    logger2.info('  Finish split the video datasets to train, test and val! ')

    # data loader
    logger2.info('  Start setting the video frames to DataLoader:')
    root_video_frames_folder = dir_video_frames
    train_fpath_label = train_file
    test_spath_label = test_file
    val_fpath_label = val_file

    logger2.info('      Root path of video frames folder : ' + str(root_video_frames_folder))
    logger2.info('      Train fpath label : ' + str(train_fpath_label))
    logger2.info('      Test fpath label : ' + str(test_spath_label))
    logger2.info('      Val fpath label : ' + str(val_fpath_label))

    # load train set
    train_loader = load_data(root_video_frames_folder, train_fpath_label, opt.batch_size, shuffle=True, num_workers=0, train=True, num_frames=opt.sample_frames_nums)
    logger2.info('      Train loader : ' + str(train_loader))
    # load test set
    test_loader = load_data(root_video_frames_folder, test_spath_label, opt.batch_size, shuffle=False, num_workers=0, train=False, num_frames=opt.sample_frames_nums)
    logger2.info('      Test loader : ' + str(test_loader))
    # load val set
    val_loader = load_data(root_video_frames_folder, val_fpath_label, opt.batch_size, shuffle=False, num_workers=0, train=False, num_frames=opt.sample_frames_nums)
    logger2.info('      Val loader : ' + str(val_loader))
    logger2.info('  Finish setting the video frames to DataLoader!')

    logger2.info('  Start create the model of C3D,resnet18,resnet34,resnet50 or resnet101 :')
    if opt.model == 'C3D':
        model = C3D(opt.hash_length, opt.class_number, pretrained=True)
    elif opt.model == 'resnet18':
        model = Res3D_Hash_Model(18, opt.hash_length, opt.class_number, pretrained=True)
    elif opt.model == 'resnet34':
        model = Res3D_Hash_Model(34, opt.hash_length, opt.class_number, pretrained=True)
    model.to(opt.device)
    logger2.info('      model : {}'.format(model))
    logger2.info('  Finish created the model of {}!'.format(opt.model))

    logger2.info('  Start create the triplet_loss and optimizer and scheduler:')
    triplet_loss = TripletLoss(opt.margin).to(opt.device)
    cross_entropy_loss = torch.nn.CrossEntropyLoss().to(opt.device)
    average_precision_loss = AveragePrecisionLoss().to(opt.device)
    optimizer = torch.optim.SGD(model.parameters(), lr=opt.learning_rate, momentum=0.9, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, opt.step_lr)
    logger2.info('  Finish create the triplet_loss and optimizer and scheduler!')

    # =================================================================================================================
    logger2.info('  Start create save result path and setting:')
    runs = sorted(glob.glob(os.path.join('run', 'run_*')))
    run_id = int(runs[-1].split('_')[-1]) if runs else 0
    save_dir = os.path.join('run', 'run_' + str(run_id+1))

    log_dir = os.path.join(save_dir, 'models', datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
    logger2.info('log_dir : ' + str(log_dir))
    writer = SummaryWriter(log_dir=log_dir)
    logger2.info('  Finish create save result path and setting!')
    # =================================================================================================================

    logger2.info('  Start Training:')
    total_train_step = len(train_loader)
    total_val__step = len(val_loader)
    print('total_train_step : ' + str(total_train_step))
    print('total_val__step : ' + str(total_val__step))

    for epoch in range(opt.num_epochs):
        start_time = time.time()
        scheduler.step()
        running_loss = 0.0
        """
        step: 0 ~ 299
        frames.shape = torch.Size([32, 3, 16, 112, 112])
        label.shape = torch.Size([32])
        """
        logger2.info('      Try to Start the training in epoch:{}'.format(epoch+1))
        for step, (frames, labels) in enumerate(tqdm(train_loader)):

            frames = Variable(frames, requires_grad=True).to(opt.device)
            labels = Variable(labels).to(opt.device)

            # hash_features : [ batch_size ,hash_length ] : [64, 512]
            hash_features, class_features = model(frames)

            loss_01 = triplet_loss(hash_features, labels)
            loss_02 = cross_entropy_loss(class_features, labels)
            loss_03 = average_precision_loss(hash_features, labels)

            loss = 1*loss_01 + 0.1*loss_02 + 0.1*loss_03

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * frames.size(0)

        avg_epoch_loss = running_loss / total_train_step    # the average loss of every epoch
        writer.add_scalar('data/train_loss_epoch', avg_epoch_loss, epoch)
        stop_time = time.time()
        logger2.info("      Finish the [Train] in Epoch : {}/{}, Execution time:{}, Loss: {}".format(epoch + 1, opt.num_epochs, str(stop_time - start_time), avg_epoch_loss))

        if epoch % 20 == 0:   # % opt.save_model_every_n_epoch == 0:
            logger2.info('      Try to save the model in epoch:{}'.format(epoch))
            save_model_path = os.path.join(save_dir, 'models', opt.model + '_' + opt.video_dataset + '_epoch_' + str(epoch) + '.pth.tar')
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'opt_dict': optimizer.state_dict(),
            }, save_model_path)
            logger2.info("      Save model at {}\n".format(os.path.join(save_dir, 'models', opt.model + '_' + opt.video_dataset + '_epoch_' + str(epoch) + '.pth.tar')))

        # compute the mAP every 20 epochs
        if epoch % 20 == 0:
            with torch.no_grad():
                logger2.info('      Try to compute the mAP in epoch:{}'.format(epoch+1))
                model.eval()
                start_time = time.time()
                MAP = compute_mAP.test_MAP(model=model, train_dataloader=train_loader, test_loader=test_loader)
                writer.add_scalar('data/test_mAP_epoch', MAP, epoch)
                stop_time = time.time()
                logger2.info('      Finish to compute the mAP, Execution time:{} mAP:{}'.format(str(stop_time - start_time), MAP))

    logger2.info('  Finish Training!')
    writer.close()

if __name__ == '__main__':

    logger2.info("================================================================================")
    logger2.info("This is main function!")

    opt = get_opt()
    logger2.info('User optional : ' + str(opt))

    opt.device = torch.device('cpu' if opt.no_cuda else 'cuda')
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger2.info('Choose the device : ' + str(opt.device))

    index = torch.cuda.device_count()
    logger2.info('The numbers of cuda : ' + str(index))

    main_worker(opt)







