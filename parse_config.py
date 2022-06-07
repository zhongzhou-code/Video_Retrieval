import logging

import argparse
from pathlib import Path

logger = logging.getLogger(__name__)

def parse_opts():
    parser = argparse.ArgumentParser(description="My_Video_Retrieval Config")
    # parser.add_argument('--root_path', default=None, type=Path, help='Root directory path')

    # Which dataset is selected
    parser.add_argument('--video_dataset', default="ucf101", type=str,
                        help='Used dataset (kinetics | ucf101 | hmdb51 | jhmdb | activitynet)')

    parser.add_argument('--class_number', default=101, type=int,
                        help='Used dataset labels number 101 | 51 | 400')

    # Which operation to perform, train or test or validate
    # Configure input sample size and quantity
    parser.add_argument('--sample_frames_nums', default=16, type=int, help='The number of frames of inputs')    # optional change #

    parser.add_argument('--sample_size', default=112, type=int, help='Height and Width of inputs')

    parser.add_argument('--batch_size', default=8, type=int, help='The number of  batch of inputs')  #change#8 16 32 64

    # Which model is selected for training
    parser.add_argument('--model', default='C3D', type=str,
                        help='(C3D | resnet18 | resnet34 | resnet50 | resnet101 | resnet152 | resnet200 | )')   # optional

    parser.add_argument('--pretrain_path', default=None, type=Path, help='Pretrained model path (.pth).')

    parser.add_argument('--resnet_shortcut', default='A', type=str, help='Shortcut type of resnet (A | B)')

    # Training
    parser.add_argument('--no_cuda', action='store_true', help='If true, cuda is not used.')

    parser.add_argument('--num_epochs', default=100, type=int, help='Number of total epochs to run')

    parser.add_argument('--learning_rate', default=0.001, type=float,
                        help='Initial learning rate (divided by 10 while training by lr scheduler)')

    parser.add_argument('--optimizer', default='sgd', type=str, help='Currently only support SGD')

    parser.add_argument('--step_lr', type=int, default=20, help='change lr per strp_lr epoch')

    # Configure output sample size and quantity
    parser.add_argument('--hash_length', type=int, default=512, help='the hash code length')

    parser.add_argument('--margin', type=float, default=0.3, help='triplet loss margin')

    # Output tensorboard log file.
    parser.add_argument('--tensorboard', action='store_true', help='If true, output tensorboard log file.')

    args = parser.parse_args()

    return args