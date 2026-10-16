#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
University of Sherbrooke
Date:
Authors: Hassan Kajila & Pierre-Marc Jodoin
License: Opensource, free to use
Other: Suggestions are welcome
"""

import argparse
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets
import numpy as np
import os
import time
# Import models and utilities
# from src.utils.preprocess import get_data_loader, get_data_loader_svhn
# from src.utils.utils import optimizer_setup, create_figure
from src.models.CNNVanilla import CnnVanilla
from src.models.AlexNet import AlexNet
from src.models.VggNet import VggNet
from src.models.ResNet import ResNet
from src.models.UNet import UNet
from src.models.yourUNet import YourUNet
from src.models.yourSegNet import YourSegNet
from src.manage.HDF5Dataset import HDF5Dataset
from src.manage.DataManager import DataManager
from src.manage.CNNTrainTestManager import CNNTrainTestManager, optimizer_setup
from src.utils.losses import FocalLoss, DiceLoss
from src.utils.data_augmentation import ACDCJointTransform

# verify if GPU is available and print the number of GPUs available
print("=" * 50)
use_cuda = torch.cuda.is_available()
print(f"CUDA available: {use_cuda}")
if use_cuda:
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print(f"GPU used: {torch.cuda.get_device_name(0)}")
    print(f"Index of active GPU: {torch.cuda.current_device()}")
    # Reserve GPU memory to avoid fragmentation
    torch.backends.cudnn.benchmark = True
else:
    print("ATTENTION: No GPU detected! Training will be performed on CPU.")
print("=" * 50)


def argument_parser():
    """
        A parser to allow user to easily experiment different models along with datasets and differents parameters
    """
    parser = argparse.ArgumentParser(usage='\n python3 train.py [model] [dataset] [hyper_parameters]'
                                           '\n python3 train.py --model UNet [hyper_parameters]'
                                           '\n python3 train.py --model UNet --predict',
                                     description="This program allows to train different models of classification on"
                                                 " different datasets. Be aware that when using UNet model there is no"
                                                 " need to provide a dataset since UNet model only train "
                                                 "on acdc dataset.",
                                     add_help=True)
    parser.add_argument('--model', type=str, default="CnnVanilla",
                        choices=["CnnVanilla", "VggNet", "AlexNet", "ResNet", "YourUNet", "YourSegNet", "UNet"])
    parser.add_argument('--dataset', type=str, default="cifar10", choices=["cifar10", "svhn", "acdc"])
    parser.add_argument('--loss', type=str, default="CE", choices=["CE", "Dice", "Focal"],
                        help="Loss function to use (CE: Cross Entropy, Dice: Dice Loss, Focal: Focal Loss)")
    parser.add_argument('--batch_size', type=int, default=20,
                        help='The size of the training batch')
    parser.add_argument('--optimizer', type=str, default="Adam", choices=["Adam", "SGD"],
                        help="The optimizer to use for training the model")
    parser.add_argument('--num-epochs', type=int, default=10,
                        help='The number of epochs')
    parser.add_argument('--validation', type=float, default=0.1,
                        help='Percentage of training data to use for validation')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--data_aug', action='store_true',
                        help="Activate basic data augmentation")
    parser.add_argument('--rotation', action='store_true',
                        help="Activate random rotation for data augmentation")
    parser.add_argument('--elastic', action='store_true',
                        help="Activate elastic deformation for data augmentation")
    parser.add_argument('--flip_crop', action='store_true',
                        help="Activate elastic deformation for data augmentation")
    parser.add_argument('--predict', type=str,
                        help="Name of the file containing model weights used to make "
                             "segmentation prediction on test data")
    parser.add_argument('--max_samples', type=int, default=None,
                        help="Maximum number of training samples to use (for quick testing)")
    parser.add_argument('--visualize_augmentation', action='store_true',
                        help="Visualize data augmentation effects on sample images")
    return parser.parse_args()


def display_time(elapsed_time):
    """
    Displays the elapsed time in an appropriate format (seconds, minutes, or hours).
    """
    print("")
    if elapsed_time < 60:
        print(f"Total training time: {elapsed_time:.2f} seconds")
    elif elapsed_time < 3600:
        elapsed_time /= 60
        print(f"Total training time: {elapsed_time:.2f} minutes")
    else:
        elapsed_time /= 3600
        print(f"Total training time: {elapsed_time:.2f} hours")


if __name__ == "__main__":

    args = argument_parser()

    batch_size = args.batch_size
    num_epochs = args.num_epochs
    val_set = args.validation
    learning_rate = args.lr
    data_augment = args.data_aug
    use_rotation = args.rotation
    use_elastic = args.elastic
    use_flip_crop = args.flip_crop

    # Display activated augmentation options
    if data_augment:
        print('Basic data augmentation enabled!')
    if use_rotation:
        print('Random rotation augmentation enabled!')
    if use_elastic:
        print('Elastic deformation augmentation enabled!')
    if not (data_augment or use_rotation or use_elastic):
        print('No data augmentation enabled!')

    # Set the HDF5 path according to your file location
    acdc_hdf5_file = 'data/ift780_acdc.hdf5'

    # Prepare ACDC transforms
    acdc_transforms = []

    if use_rotation or use_elastic:
        def transform_func(x):
            # Apply augmentation transforms
            transform = ACDCJointTransform(
                rotation=use_rotation,
                elastic=use_elastic,
                rotation_degrees=60,
                elastic_alpha=50.0,
                elastic_sigma=5.0
            )

            result = transform(x)

            # Ensure the correct format for PyTorch (NCHW)
            if isinstance(result, torch.Tensor):
                if len(result.shape) == 2:  # (H, W)
                    result = result.unsqueeze(0)  # Add channel dimension -> (1, H, W)
                elif len(result.shape) == 3 and result.shape[2] <= 4:  # (H, W, C)
                    result = result.permute(2, 0, 1)  # -> (C, H, W)
            else:  # Numpy array
                if len(result.shape) == 2:  # (H, W)
                    result = np.expand_dims(result, 0)  # -> (1, H, W)
                    result = torch.from_numpy(result).float()
                elif len(result.shape) == 3 and result.shape[2] <= 4:  # (H, W, C)
                    result = np.transpose(result, (2, 0, 1))  # -> (C, H, W)
                    result = torch.from_numpy(result).float()

            return result

        acdc_base_transform = transform_func

    elif use_flip_crop:
        flip_crop_transform = transforms.Compose(
            [
                transforms.RandomCrop(200, padding=4),
                transforms.RandomAffine(10),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(degrees=15),
            ]
        )
        acdc_base_transform = flip_crop_transform
    else:
        # Otherwise, use tensor conversion only
        acdc_base_transform = transforms.ToTensor()

    # Transformations for CIFAR10/SVHN
    data_augment_transform = transforms.Compose(
        [
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        ]
    )
    base_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    transform = base_transform
    if data_augment:
        transform = transforms.Compose([data_augment_transform, transform])
    data_path = 'data'
    if args.dataset == 'cifar10':
        # Download the train and test set and apply transform on it
        train_set = datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform)
        test_set = datasets.CIFAR10(root=data_path, train=False, download=True, transform=base_transform)

    elif args.dataset == 'svhn':
        # Download the train and test set and apply transform on it
        train_set = datasets.SVHN(root=data_path, split='train', download=True, transform=transform)
        test_set = datasets.SVHN(root=data_path, split='test', download=True, transform=base_transform)

    if args.optimizer == 'SGD':
        optimizer_factory = optimizer_setup(torch.optim.SGD, lr=learning_rate, momentum=0.9)
    elif args.optimizer == 'Adam':
        optimizer_factory = optimizer_setup(optim.Adam, lr=learning_rate)

    # Select the loss function
    if args.loss == "CE":
        loss_fn = nn.CrossEntropyLoss()
    elif args.loss == "Dice":
        loss_fn = DiceLoss()
    elif args.loss == "Focal":
        loss_fn = FocalLoss()

    if args.model == 'CnnVanilla':
        model = CnnVanilla(num_classes=10)
    elif args.model == 'AlexNet':
        model = AlexNet(num_classes=10)
    elif args.model == 'VggNet':
        model = VggNet(num_classes=10)
    elif args.model == 'ResNet':
        model = ResNet(num_classes=10)
    elif args.model == 'YourUNet':
        model = YourUNet(num_classes=4)
        args.dataset = 'acdc'
        # Create the dataset with augmentation settings
        train_set = HDF5Dataset('train', acdc_hdf5_file, transform=acdc_base_transform)
        test_set = HDF5Dataset('test', acdc_hdf5_file, transform=acdc_base_transform)
        if args.max_samples is not None:
            print(f"Using {args.max_samples} samples for training")
            train_set = torch.utils.data.Subset(train_set, range(min(args.max_samples, len(train_set))))
    elif args.model == 'YourSegNet':
        model = YourSegNet(num_classes=4)
        args.dataset = 'acdc'
        # Create the dataset with augmentation settings
        train_set = HDF5Dataset('train', acdc_hdf5_file, transform=acdc_base_transform)
        test_set = HDF5Dataset('test', acdc_hdf5_file, transform=acdc_base_transform)
        if args.max_samples is not None:
            print(f"Using {args.max_samples} samples for training")
            train_set = torch.utils.data.Subset(train_set, range(min(args.max_samples, len(train_set))))
    elif args.model == 'UNet':
        model = UNet(num_classes=4)
        args.dataset = 'acdc'
        # Create the dataset with augmentation settings
        train_set = HDF5Dataset('train', acdc_hdf5_file, transform=acdc_base_transform)
        test_set = HDF5Dataset('test', acdc_hdf5_file, transform=acdc_base_transform)

    # Create a model-specific checkpoint directory
    checkpoint_dir = f'checkpoints/{args.model}_{args.loss}'
    if args.max_samples is not None:
        checkpoint_dir += f'_samples{args.max_samples}'
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Checkpoints will be saved in: {checkpoint_dir}")

    model_trainer = CNNTrainTestManager(model=model,
                                        trainset=train_set,
                                        testset=test_set,
                                        batch_size=batch_size,
                                        loss_fn=loss_fn,
                                        optimizer_factory=optimizer_factory,
                                        validation=val_set,
                                        use_cuda=use_cuda,
                                        checkpoint_dir=checkpoint_dir)

    if args.predict is not None:
        # Load checkpoint
        print(f"Loading checkpoint: {args.predict}")
        checkpoint = torch.load(args.predict, map_location=torch.device('cuda' if use_cuda else 'cpu'))

        # Check whether the checkpoint contains model_state_dict (new format) or weights directly (old format)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Validation accuracy of loaded model: {checkpoint.get('val_acc', 'N/A')}")
        else:
            model.load_state_dict(checkpoint)

        print("Predicting masks for random images in the test set")
        model_trainer.plot_image_mask_prediction()
    elif args.visualize_augmentation:
        # Visualize the effects of data augmentation
        print("Visualizing the effects of data augmentation...")
        model_trainer.visualize_data_augmentation(num_samples=3)
    else:
        # Display GPU usage information
        if use_cuda:
            print(
                f"Training {model.__class__.__name__} on {args.dataset} for {args.num_epochs} epochs using GPU: {torch.cuda.get_device_name(0)}")
        else:
            print(
                f"Training {model.__class__.__name__} on {args.dataset} for {args.num_epochs} epochs using CPU")

        start_time = time.time()

        model_trainer.train(num_epochs)

        # Display final metrics
        final_train_acc = model_trainer.metric_values['train_acc'][-1]
        final_val_acc = model_trainer.metric_values['val_acc'][-1]
        final_train_loss = model_trainer.metric_values['train_loss'][-1]
        final_val_loss = model_trainer.metric_values['val_loss'][-1]

        print("\n" + "=" * 30 + " FINAL RESULTS " + "=" * 30)
        print(f"Final training accuracy: {final_train_acc:.4f}")
        print(f"Final validation accuracy: {final_val_acc:.4f}")
        print(f"Final training loss: {final_train_loss:.4f}")
        print(f"Final validation loss: {final_val_loss:.4f}")
        print("=" * 77 + "\n")

        model_trainer.evaluate_on_test_set()

        display_time(time.time() - start_time)

        if isinstance(model, UNet):
            model.save()  # save the model's weights for prediction (see help for more details)
            model_trainer.plot_image_mask_prediction()
        model_trainer.plot_metrics()
