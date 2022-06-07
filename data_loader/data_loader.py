import os
import cv2
import numpy as np
import torch
import torch.utils.data as data

from PIL import Image

from utils_log.perf_log import logger2

class CustomDataset(data.Dataset):
    """
        root_folder: data_frames/16-ucf101
        fpath_label: data_label/16-ucf101/train.txt
        transform:
        num_frames: 16 / 32 / 64
    """
    def __init__(self, fpath_label, transform=None, num_frames=16):
        logger2.info('          Come into the function of CustomDataset(data.Dataset) : ')
        with open(fpath_label) as file:
            list_frames_label_pair = file.readlines()
        file.close()
        # list_label: 'WritingOnBoard/v_WritingOnBoard_g13_c02.avi 100\n', 'WritingOnBoard/v_WritingOnBoard_g13_c03.avi 100\n'
        list_frames_path = []
        list_labels = []
        for item in list_frames_label_pair:
            path_frames = item.split(' ')[0]
            # print(path_frames)
            if os.path.exists(path_frames):
                list_frames_path.append(path_frames)
                list_labels.append(item.split(' ')[1].replace('\n', ''))
            else:
                exit(0)

        # print(list_labels)
        # print('len(list_labels): ' + str(len(list_labels)))
        # print('len(list_frames_path): ' + str(len(list_frames_path)))
        # print(list_frames_path[0])

        #
        self.list_frames_path = list_frames_path
        #
        self.list_labels = list_labels

        #
        self.frames_size = len(list_frames_path)
        #
        self.label_size = len(self.list_labels)

        #
        self.transform = transform
        #
        self.num_frames = num_frames


    def __getitem__(self, index):

        #
        if index < self.label_size:

            label = self.list_labels[index]
            frames_dir = self.list_frames_path[index]

        else:
            exit(0)
        #
        list_video_frames_images = os.listdir(frames_dir)
        #
        frames_length = self.num_frames

        #
        assert len(list_video_frames_images) == self.num_frames, {frames_dir}

        #  16*3*112*112
        frames_array = np.zeros((frames_length, 3, 112, 112), dtype=np.float32)

        for i in range(frames_length):
            #
            frame = Image.open(frames_dir + '/' + list_video_frames_images[i]).convert("RGB")
            #
            if not self.transform == None:
                frame = self.transform(frame)
                frame = frame.numpy()
            # frames_array.shape = [16, 3, 112, 112]
            frames_array[i, :, :, :] = frame

        # frames_array.shape = [3, 16, 112, 112]
        frames_array = frames_array.transpose((1, 0, 2, 3))

        # label.shape = [1]
        label = torch.tensor(int(label))
        # frames.shape = [3, 16, 112, 112]
        frames = torch.tensor(frames_array)

        return frames, label

    def __len__(self):
        return len(self.list_frames_path)