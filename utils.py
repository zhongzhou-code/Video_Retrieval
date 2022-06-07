import os
import cv2
import numpy as np
import torchvision.transforms as transforms
from data_loader.data_loader import CustomDataset
import torch.utils.data as data

from utils_log.perf_log import logger2

"""
dataset_path = data/ucf101
dataset_frames_save_path = data_frames/16-ucf101
"""
def video2frames(dataset_path, dataset_frames_save_path, save_frames_nums):
    logger2.info('      Come into the function of video2frames')
    """
    :param video_path:
    :param frames_path:
    :return:
    """
    list_of_video_class_name = os.listdir(dataset_path)
    logger2.info('          List of videos class names : '+str(list_of_video_class_name))

    list_of_video_class_names_path = []
    for video_class_name in list_of_video_class_name:
        list_of_video_class_names_path.append(dataset_path + '/' + video_class_name)
    logger2.info('          List of videos class names path : ' + str(list_of_video_class_names_path))

    save_video_label_pairs_path = dataset_frames_save_path + ".txt"
    logger2.info('          The path of save video and label pair : ' + str(save_video_label_pairs_path))

    class_count = 0
    """
        Retrieval all videos
    """
    for video_classes_name_path in list_of_video_class_names_path:

        list_of_video_name = os.listdir(video_classes_name_path)

        list_of_video_name_path = []

        for video_name in list_of_video_name:
            list_of_video_name_path.append(video_classes_name_path + '/' + video_name)
        logger2.info('          The List of video path: ' + str(list_of_video_name_path))

        for video_path in list_of_video_name_path:
            # logger2.info('              Try to extract video frames from : ' + str(video_path))
            save_video_frames_path = dataset_frames_save_path + '/' + video_classes_name_path.split('/')[-1] + '/' + video_path.split('/')[-1].split('.')[0]

            if os.path.exists(save_video_frames_path):
                # logger2.info('              The video frames have been saved to : ' + str(save_video_frames_path))
                continue

            print('     The video frames will be saved to : ' + str(save_video_frames_path))   # data_frames/16-hmdb51/brush_hair/Brushing_my_long_hair_brush_hair_u_nm_np1_ba_goo_2

            video_cap = cv2.VideoCapture(video_path)

            #
            the_numbers_of_video_frames = 0
            while video_cap.isOpened():

                ret, frame = video_cap.read()

                if ret:
                    the_numbers_of_video_frames += 1
                else:
                    #
                    break

            video_cap.release()

            print('     The numbers of frames for the video: ' + str(the_numbers_of_video_frames))

            #
            if the_numbers_of_video_frames <= save_frames_nums:
                print("     Error : the_numbers_of_video_frames <= save_frames_nums! video path : " + str(video_path))
                continue

            #
            video_interval_frame_time = int(the_numbers_of_video_frames / save_frames_nums)
            print('     The interval time of the video : ' + str(video_interval_frame_time))

            # frames count
            frames_count = 0
            #
            save_frames_index = 0

            video_cap = cv2.VideoCapture(video_path)

            while video_cap.isOpened():
                print(frames_count)

                ret, frame = video_cap.read()

                if ret:
                    #
                    if not os.path.exists(save_video_frames_path):
                        os.makedirs(save_video_frames_path)

                    if (frames_count % video_interval_frame_time == 0) and (save_frames_index < save_frames_nums):
                        #
                        retval = cv2.imwrite(save_video_frames_path + '/' + str(int(save_frames_index)).zfill(4) + ".jpg", frame)
                        print(frames_count, frames_count % video_interval_frame_time, save_frames_index)
                        save_frames_index += 1
                        #
                        assert retval is True, {save_frames_index, save_frames_nums, save_video_frames_path}

                else:
                    break

                # frames count
                frames_count += 1

            print("     Video : " + save_video_frames_path.split('/')[-1] + ' save successful!  ' + 'Save Path : ' + save_video_frames_path + '\n')
            with open(save_video_label_pairs_path, "a+") as video_label_pairs_file:
                video_label_pairs_file.writelines(str(save_video_frames_path) + ' ' + str(class_count) + '\n')
                video_label_pairs_file.close()

            video_cap.release()

        class_count += 1

        with open(save_video_label_pairs_path, 'r') as file:
            list_01 = file.readlines()
            file.close()
        for path_label in list_01:
            path = str(path_label).split(' ')[0]
            assert save_frames_nums == len(os.listdir(path)), path
    """
    
    """

    #
    return save_video_label_pairs_path


def split_dataset(path_video_label_pairs_file, ratio_train, ratio_val, ratio_test):
    logger2.info('      Come into the function of split_dataset:')
    """
    :param path_video_label_pairs_file:
    :return:
    """
    # split_root_path : data_label/16-hmdb51/
    dir_name = str(path_video_label_pairs_file).split('/')[1].split('.')[0]
    split_root_path = 'data_label/' + dir_name
    print('split_root_path: ' + str(split_root_path))
    if not os.path.exists(split_root_path):
        os.makedirs(split_root_path)
    dataset_path = {'train_file_path': str(split_root_path + '/' + 'train.txt'),
                    'test_file_path': split_root_path + '/' + 'test.txt',
                    'val_file_path': split_root_path + '/' + 'val.txt'}
    logger2.info('          train_file_path:' + dataset_path['train_file_path'] + ' test_file_path:'+dataset_path['test_file_path'] + ' val_file_path:'+dataset_path['val_file_path'])

    if os.path.exists(dataset_path['train_file_path']) and os.path.exists(dataset_path['test_file_path']) and os.path.exists(dataset_path['val_file_path']):
        logger2.info('          All the file is exists! ')
        return dataset_path['train_file_path'], dataset_path['test_file_path'], dataset_path['val_file_path']

    logger2.info('          All the file is not exists! Try to create all split file!')
    if os.path.exists(path_video_label_pairs_file):

        with open(path_video_label_pairs_file, 'r') as file:
            list_full = file.readlines()
            print('list_full.__len__() : ' + str(list_full.__len__()))
            file.close()

        np.random.shuffle(list_full)

        test_set_length = int(len(list_full) * ratio_test)
        val_set_length = int(len(list_full) * ratio_val)
        train_set_length = len(list_full) - test_set_length - val_set_length
        print('   test_set_length: ' + str(test_set_length), '   val_set_length:' + str(val_set_length), '   train_set_length:' + str(train_set_length))


        train_list = []
        test_list = []
        val_list = []

        for i in range(int(train_set_length)):
            train_list.append(list_full[i])

        for i in range(int(train_set_length), int(train_set_length + test_set_length)):
            test_list.append(list_full[i])

        for i in range(int(train_set_length + test_set_length), int(train_set_length + test_set_length + val_set_length)):
            val_list.append(list_full[i])

        print('   test_list_length: ' + str(test_list), '   val_list_length:' + str(val_list), '   train_list_length:' + str(train_list))


        file = open(dataset_path['train_file_path'], "w")
        file.writelines(train_list)
        file.close()

        file = open(dataset_path['test_file_path'], "w")
        file.writelines(test_list)
        file.close()

        file = open(dataset_path['val_file_path'], "w")
        file.writelines(val_list)
        file.close()

    return dataset_path['train_file_path'], dataset_path['test_file_path'], dataset_path['val_file_path']


def load_data(root_video_frames_folder, fpath_label, batch_size, shuffle=True, num_workers=8, train=False, num_frames=16):
    logger2.info('      Come into the function of load_data:')
    """
        transforms.ToPILImage(), Converts a Tensor or a numpy of shape H x W x C to a PIL Image C x H x W
    """
    if train:
        transform = transforms.Compose([
            transforms.Resize((128, 171)),
            transforms.CenterCrop((112, 112)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.434, 0.405, 0.378), std=(0.152, 0.149, 0.157))])
    else:
        transform = transforms.Compose([
            transforms.Resize((128, 171)),
            transforms.CenterCrop((112, 112)),  # Center
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.434, 0.405, 0.378), std=(0.152, 0.149, 0.157))])
    logger2.info('      transform : ' + str(transform))
    data_ = CustomDataset(fpath_label=fpath_label, transform=transform, num_frames=num_frames)
    logger2.info('      data_ : ' + str(data_))

    loader_ = data.DataLoader(dataset=data_, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=True)

    return loader_
