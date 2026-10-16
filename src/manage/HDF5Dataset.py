# -*- coding:utf-8 -*-


"""
University of Sherbrooke
Authors: Hassan Kajila & Pierre-Marc Jodoin
Other: Suggestions are welcome
"""

import h5py
import random
import numpy as np
import torch
from torch.utils import data
from src.utils.utils import centered_resize


class HDF5Dataset(data.Dataset):
    """
    class that loads hdf5 dataset object
    """

    def __init__(self, set, file_path, input_size: int = 256, transform=None, num_classes=4, pin_memory=True):
        """
        Args:
            set: dataset split ('train', 'test', etc.)
            file_path: path to the HDF5 file
            input_size: size of the input images
            transform: transformations to apply
            num_classes: number of classes
            pin_memory: if True, keeps the data in pinned CPU memory to speed up transfers to the GPU
        """
        self.file_path = file_path
        self.set = set
        self.input_size = input_size
        self.transform = transform
        self.num_classes = num_classes
        self.pin_memory = pin_memory
        self.data = self._load_data()

        # Preload the patient keys
        with h5py.File(self.file_path, 'r') as f:
            self.key_list = list(f[self.set].keys())

        # Check whether CUDA is available for GPU optimizations
        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            # Let cuDNN pick the fastest convolution algorithms
            torch.backends.cudnn.benchmark = True

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

            # fix a random seed (using a smaller range to avoid int32 overflow)
            seed = np.random.randint(0, 1000000)
            torch.manual_seed(seed)
            # pass image and ground truth trough transform for data augmentation and normalization purposes
            random.seed(seed)
            img_slice = self.transform(img_slice)
            random.seed(seed)
            gt_slice = self.transform(gt_slice)
        else:
            # Convert to PyTorch tensors
            img_slice = torch.from_numpy(img_slice).float()
            gt_slice = torch.from_numpy(gt_slice).float()

        # ground truth must be encoded in categorical form instead of one_hot vector
        # in order to allow pytorch's CrossEntropy to compute the loss
        # Make sure the labels always stay in the valid range [0, num_classes - 1]
        if isinstance(gt_slice, torch.Tensor):
            gt_categorical = torch.argmax(gt_slice, dim=0)
            # Clamp out-of-range values
            gt_categorical = torch.clamp(gt_categorical, 0, self.num_classes - 1)
        else:
            gt_categorical = np.argmax(gt_slice, axis=0)
            # Clamp out-of-range values
            gt_categorical = np.clip(gt_categorical, 0, self.num_classes - 1)
            gt_categorical = torch.from_numpy(gt_categorical).long()

        # Speed up the transfer to the GPU
        if self.pin_memory and self.use_cuda:
            img_slice = img_slice.pin_memory()
            gt_categorical = gt_categorical.pin_memory()

        return img_slice, gt_categorical

    def __len__(self):
        """
        return the length of the dataset
        """
        return int(np.floor(len(self.data)))

    @staticmethod
    def get_data_slice(key, position, file):
        """
        Return one slice from the hdf5 file
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
