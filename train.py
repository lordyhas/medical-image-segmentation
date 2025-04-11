#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
University of Sherbrooke
Date:
Authors: Mamadou Mountagha BAH & Pierre-Marc Jodoin
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
# Importation des modèles et des outils
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

# Vérification et affichage des informations sur le GPU
print("=" * 50)
use_cuda = torch.cuda.is_available()
print(f"CUDA disponible: {use_cuda}")
if use_cuda:
    print(f"Nombre de GPUs: {torch.cuda.device_count()}")
    print(f"GPU utilisé: {torch.cuda.get_device_name(0)}")
    print(f"Index du GPU actif: {torch.cuda.current_device()}")
    # Réserver la mémoire GPU pour éviter les fragmentations
    torch.backends.cudnn.benchmark = True
else:
    print("ATTENTION: Aucun GPU détecté! L'entraînement sera effectué sur CPU.")
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
    Affiche le temps écoulé au format approprié (secondes, minutes ou heures).
    """
    print("")
    if elapsed_time < 60:
        print(f"Temps d'entraînement total: {elapsed_time:.2f} secondes")
    elif elapsed_time < 3600:
        elapsed_time /= 60
        print(f"Temps d'entraînement total: {elapsed_time:.2f} minutes")
    else:
        elapsed_time /= 3600
        print(f"Temps d'entraînement total: {elapsed_time:.2f} heures")


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

    # Affichage des options d'augmentation activées
    if data_augment:
        print('Augmentation de données de base activée!')
    if use_rotation:
        print('Augmentation par rotation aléatoire activée!')
    if use_elastic:
        print('Augmentation par déformation élastique activée!')
    if not (data_augment or use_rotation or use_elastic):
        print('Aucune augmentation de données activée!')

    # set hdf5 path according your hdf5 file location
    acdc_hdf5_file = 'data/ift780_acdc.hdf5'

    # Préparation des transformations pour ACDC
    acdc_transforms = []

    if use_rotation or use_elastic:
        def transform_func(x):
            # Appliquer les transformations d'augmentation
            transform = ACDCJointTransform(
                rotation=use_rotation,
                elastic=use_elastic,
                rotation_degrees=60,
                elastic_alpha=50.0,
                elastic_sigma=5.0
            )

            result = transform(x)

            # Assurer le format correct pour PyTorch (NCHW)
            if isinstance(result, torch.Tensor):
                if len(result.shape) == 2:  # (H, W)
                    result = result.unsqueeze(0)  # Ajouter dimension canal -> (1, H, W)
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
        # Sinon, utiliser seulement la conversion en tenseur
        acdc_base_transform = transforms.ToTensor()

    # Transformations pour CIFAR10/SVHN
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

    # Sélection de la fonction de perte
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
        # Créer le dataset avec les options d'augmentation
        train_set = HDF5Dataset('train', acdc_hdf5_file, transform=acdc_base_transform)
        test_set = HDF5Dataset('test', acdc_hdf5_file, transform=acdc_base_transform)
        if args.max_samples is not None:
            print(f"Utilisation de {args.max_samples} échantillons pour l'entraînement")
            train_set = torch.utils.data.Subset(train_set, range(min(args.max_samples, len(train_set))))
    elif args.model == 'YourSegNet':
        model = YourSegNet(num_classes=4)
        args.dataset = 'acdc'
        # Créer le dataset avec les options d'augmentation
        train_set = HDF5Dataset('train', acdc_hdf5_file, transform=acdc_base_transform)
        test_set = HDF5Dataset('test', acdc_hdf5_file, transform=acdc_base_transform)
        if args.max_samples is not None:
            print(f"Utilisation de {args.max_samples} échantillons pour l'entraînement")
            train_set = torch.utils.data.Subset(train_set, range(min(args.max_samples, len(train_set))))
    elif args.model == 'UNet':
        model = UNet(num_classes=4)
        args.dataset = 'acdc'
        # Créer le dataset avec les options d'augmentation
        train_set = HDF5Dataset('train', acdc_hdf5_file, transform=acdc_base_transform)
        test_set = HDF5Dataset('test', acdc_hdf5_file, transform=acdc_base_transform)

    # Créer un répertoire de checkpoint spécifique au modèle
    checkpoint_dir = f'checkpoints/{args.model}_{args.loss}'
    if args.max_samples is not None:
        checkpoint_dir += f'_samples{args.max_samples}'
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Les checkpoints seront sauvegardés dans: {checkpoint_dir}")

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
        # Chargement du checkpoint
        print(f"Chargement du checkpoint: {args.predict}")
        checkpoint = torch.load(args.predict, map_location=torch.device('cuda' if use_cuda else 'cpu'))

        # Vérifier si le checkpoint contient model_state_dict (nouveau format) ou directement les poids (ancien format)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Précision de validation du modèle chargé: {checkpoint.get('val_acc', 'N/A')}")
        else:
            model.load_state_dict(checkpoint)

        print("Prédiction de masques pour des images aléatoires du jeu de test")
        model_trainer.plot_image_mask_prediction()
    elif args.visualize_augmentation:
        # Visualiser les effets de l'augmentation de données
        print("Visualisation des effets de l'augmentation de données...")
        model_trainer.visualize_data_augmentation(num_samples=3)
    else:
        # Afficher des informations sur l'utilisation du GPU
        if use_cuda:
            print(
                f"Entraînement de {model.__class__.__name__} sur {args.dataset} pour {args.num_epochs} époques utilisant GPU: {torch.cuda.get_device_name(0)}")
        else:
            print(
                f"Entraînement de {model.__class__.__name__} sur {args.dataset} pour {args.num_epochs} époques utilisant CPU")

        start_time = time.time()

        model_trainer.train(num_epochs)

        # Afficher les métriques finales
        final_train_acc = model_trainer.metric_values['train_acc'][-1]
        final_val_acc = model_trainer.metric_values['val_acc'][-1]
        final_train_loss = model_trainer.metric_values['train_loss'][-1]
        final_val_loss = model_trainer.metric_values['val_loss'][-1]

        print("\n" + "=" * 30 + " RÉSULTATS FINAUX " + "=" * 30)
        print(f"Accuracy d'entraînement finale: {final_train_acc:.4f}")
        print(f"Accuracy de validation finale: {final_val_acc:.4f}")
        print(f"Loss d'entraînement finale: {final_train_loss:.4f}")
        print(f"Loss de validation finale: {final_val_loss:.4f}")
        print("=" * 77 + "\n")

        model_trainer.evaluate_on_test_set()

        display_time(time.time() - start_time)

        if isinstance(model, UNet):
            model.save()  # save the model's weights for prediction (see help for more details)
            model_trainer.plot_image_mask_prediction()
        model_trainer.plot_metrics()
