# -*- coding:utf-8 -*-


"""
University of Sherbrooke
Date:
Authors: Mamadou Mountagha BAH & Pierre-Marc Jodoin
License:
Other: Suggestions are welcome
"""

import h5py
import random
import numpy as np
import torch
from torch.utils import data
from utils.utils import centered_resize


class HDF5Dataset(data.Dataset):
    """
    class that loads hdf5 dataset object
    """

    def __init__(self, set, file_path, input_size: int = 256, transform=None):
        """
        Args:
        """
        self.file_path = file_path
        self.set = set
        self.input_size = input_size
        self.transform = transform
        self.data = self._load_data()

        with h5py.File(self.file_path, 'r') as f:
            self.key_list = list(f[self.set].keys())

    def _load_data(self):
        """ List of images from the HDF5 file and save them into a set list.

        :return
            list containing path to files for the corresponding set
        """

        def get_set_list(set_key, file):
            set_list = []
            f_set = file[set_key]
            for key in list(f_set.keys()):
                patient = f_set[key]
                img = patient['img']
                gt = patient['gt']

                assert img.shape[0] == gt.shape[0]

                for i in range(img.shape[0]):
                    k = '{}/{}'.format(set_key, key)
                    set_list.append((k, i))
            return set_list

        with h5py.File(self.file_path, 'r') as f:
            set_list = get_set_list(self.set, f)
            if self.set == 'train':
                set_list = np.random.permutation(set_list)
        return set_list

    def __getitem__(self, index):

        """This method loads, transforms and returns slice corresponding to the corresponding index.
        :arg
            index: the index of the slice within patient data
        :return
            A tuple (input, target)

        """
        key, position = self.data[index]

        with h5py.File(self.file_path, 'r') as f:
            img_slice, gt_slice = self.get_data_slice(key, position, f)

        # Resize the image and the ground truth
        img_slice = centered_resize(img_slice, (self.input_size, self.input_size))
        gt_slice = centered_resize(gt_slice, (self.input_size, self.input_size))

        # Need to redo the background class due to resize
        # that set the image border to 0
        summed = np.clip(gt_slice[:, :, 1:].sum(axis=-1), 0, 1)
        gt_slice[:, :, 0] = np.abs(1 - summed)

        if self.transform is not None:
            # transform numpy array using the provided transform
            # To apply the same transformations (RandomFlip, RandomCrop) on both image and ground truth
            # we need to fix the seed to make those functions return same value when using random

            # fix a random seed
            seed = np.random.randint(0, 2 ** 32)
            # print(seed)
            torch.manual_seed(seed)
            # pass image and ground truth trough transform for data augmentation and normalization purposes
            random.seed(seed)
            img_slice = self.transform(img_slice)
            random.seed(seed)
            gt_slice = self.transform(gt_slice)
        else:
            img_slice = torch.from_numpy(img_slice)
            gt_slice = torch.from_numpy(gt_slice)
        # ground truth must be encoded in categorical form instead of one_hot vector
        # in order to allow pytorch's CrossEntropy to compute the loss
        return img_slice, np.argmax(gt_slice, axis=0)

    def __len__(self):
        """
        return the length of the dadaist
        """
        return int(np.floor(len(self.data)))

    @staticmethod
    def get_data_slice(key, position, file):
        """
        Return one slice from the hdf3 file
        Args:
            key: key corresponding to each patient
            position: image and ground truth positions into patient's images
            file: the hdf5 dataset
        :return
            tuple corresponding to a slice and its corresponding ground truth
        """
        img = np.array(file['{}/img'.format(key)])
        gt = np.array(file['{}/gt'.format(key)])
        return img[int(position)], gt[int(position)]
