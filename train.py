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
import scipy.ndimage
import h5py
import matplotlib.pyplot as plt

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
from src.utils.utils import convert_mask_to_rgb_image
from src.utils.elastic_transform import ElasticTransform

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
                        help="Data augmentation")
    parser.add_argument('--predict', type=str,
                        help="Name of the file containing model weights used to make "
                             "segmentation prediction on test data")
    parser.add_argument('--max_samples', type=int, default=None,
                        help="Maximum number of training samples to use (for quick testing)")
    parser.add_argument('--data_aug_medical', action='store_true',
                        help="Apply medical image data augmentation (rotations, intensity variations)")
    parser.add_argument('--data_aug_elastic', action='store_true',
                        help="Apply elastic deformations data augmentation")
    parser.add_argument('--visualize_augmentation', action='store_true',
                        help="Visualize the effects of data augmentation on a few samples")
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
    data_aug_medical = args.data_aug_medical
    data_aug_elastic = args.data_aug_elastic

    if data_augment:
        print('Data augmentation activated!')
    if data_aug_medical:
        print('Medical image augmentation activated!')
    if data_aug_elastic:
        print('Elastic deformation augmentation activated!')

    # set hdf5 path according your hdf5 file location
    acdc_hdf5_file = 'data/ift780_acdc.hdf5'

    # Transform is used to normalize data among others
    acdc_base_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # Augmentation pour les images médicales
    medical_transform = []
    if data_aug_medical:
        medical_transform = [
            # Rotation aléatoire légère (-15 à +15 degrés)
            transforms.RandomRotation(15, fill=0),
            # Variation d'intensité aléatoire
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            # Flips horizontaux aléatoires
            transforms.RandomHorizontalFlip(),
        ]

    elastic_transform = []
    if data_aug_elastic:
        try:
            elastic_transform = [ElasticTransform()]
            print("Elastic transform successfully initialized")
        except ImportError:
            print("Warning: scipy not found, elastic transform disabled")
            data_aug_elastic = False

    # Construire les transformations pour les images médicales
    acdc_transform = acdc_base_transform
    if data_aug_medical or data_aug_elastic:
        all_transforms = medical_transform + elastic_transform + [transforms.ToTensor()]
        acdc_transform = transforms.Compose(all_transforms)
        print("Medical data augmentation pipeline created")

    base_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    transform = base_transform
    if data_augment:
        transform = transforms.Compose([acdc_transform, transform])
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
        # Utiliser la transformation avec augmentation si activée
        transform_to_use = acdc_transform if (data_aug_medical or data_aug_elastic) else acdc_base_transform
        train_set = HDF5Dataset('train', acdc_hdf5_file, transform=transform_to_use)
        test_set = HDF5Dataset('test', acdc_hdf5_file, transform=acdc_base_transform)
        if args.max_samples is not None:
            print(f"Utilisation de {args.max_samples} échantillons pour l'entraînement")
            train_set = torch.utils.data.Subset(train_set, range(min(args.max_samples, len(train_set))))
    elif args.model == 'YourSegNet':
        model = YourSegNet(num_classes=4)
        args.dataset = 'acdc'
        # Utiliser la transformation avec augmentation si activée
        transform_to_use = acdc_transform if (data_aug_medical or data_aug_elastic) else acdc_base_transform
        train_set = HDF5Dataset('train', acdc_hdf5_file, transform=transform_to_use)
        test_set = HDF5Dataset('test', acdc_hdf5_file, transform=acdc_base_transform)
        if args.max_samples is not None:
            print(f"Utilisation de {args.max_samples} échantillons pour l'entraînement")
            train_set = torch.utils.data.Subset(train_set, range(min(args.max_samples, len(train_set))))
    elif args.model == 'UNet':
        model = UNet(num_classes=4)
        args.dataset = 'acdc'
        # Utiliser la transformation avec augmentation si activée
        transform_to_use = acdc_transform if (data_aug_medical or data_aug_elastic) else acdc_base_transform
        train_set = HDF5Dataset('train', acdc_hdf5_file, transform=transform_to_use)
        test_set = HDF5Dataset('test', acdc_hdf5_file, transform=acdc_base_transform)

    # Créer un répertoire de checkpoint spécifique au modèle
    checkpoint_dir = f'checkpoints/{args.model}_{args.loss}'
    if args.max_samples is not None:
        checkpoint_dir += f'_samples{args.max_samples}'
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Les checkpoints seront sauvegardés dans: {checkpoint_dir}")

    # Visualiser l'augmentation de données si demandé
    if args.visualize_augmentation and args.dataset == 'acdc':
        # Créer un dossier pour sauvegarder les visualisations
        aug_viz_dir = 'augmentation_examples'
        os.makedirs(aug_viz_dir, exist_ok=True)

        print(f"Génération d'exemples d'augmentation de données dans {aug_viz_dir}...")

        with h5py.File(acdc_hdf5_file, 'r') as f:
            # Prendre quelques exemples aléatoires
            set_list = list(f['train'].keys())
            for idx in range(3):  # Générer 3 exemples
                # Sélectionner un patient et une coupe aléatoire
                patient_key = np.random.choice(set_list)
                patient = f['train'][patient_key]
                img = np.array(patient['img'])
                gt = np.array(patient['gt'])
                slice_idx = np.random.randint(0, img.shape[0])

                # Image et masque originaux
                orig_img = img[slice_idx]
                orig_gt = gt[slice_idx]
                orig_gt_rgb = convert_mask_to_rgb_image(np.argmax(orig_gt, axis=-1))

                # Appliquer les transformations médicales
                if data_aug_medical:
                    # Convertir en tenseur pour les transformations
                    img_tensor = torch.from_numpy(orig_img).unsqueeze(0).float()  # Ajouter dim canal
                    gt_tensor = torch.from_numpy(orig_gt).permute(2, 0, 1).float()  # HWC -> CHW

                    # Appliquer les mêmes transformations avec la même seed
                    seed = np.random.randint(0, 1000000)
                    torch.manual_seed(seed)
                    transformed_img = transforms.RandomRotation(15, fill=0)(img_tensor)
                    transformed_img = transforms.ColorJitter(brightness=0.15, contrast=0.15)(transformed_img)
                    torch.manual_seed(seed)
                    transformed_gt = transforms.RandomRotation(15, fill=0)(gt_tensor)

                    # Flip séparément pour la visualisation
                    torch.manual_seed(seed + 1)  # Nouvelle seed pour avoir un exemple différent
                    flipped_img = transforms.RandomHorizontalFlip(p=1.0)(img_tensor)
                    torch.manual_seed(seed + 1)
                    flipped_gt = transforms.RandomHorizontalFlip(p=1.0)(gt_tensor)

                    # Convertir en numpy pour l'affichage
                    transformed_img_np = transformed_img.squeeze(0).numpy()
                    transformed_gt_np = transformed_gt.argmax(dim=0).numpy()
                    transformed_gt_rgb = convert_mask_to_rgb_image(transformed_gt_np)

                    flipped_img_np = flipped_img.squeeze(0).numpy()
                    flipped_gt_np = flipped_gt.argmax(dim=0).numpy()
                    flipped_gt_rgb = convert_mask_to_rgb_image(flipped_gt_np)

                    # Créer la figure
                    fig, axs = plt.subplots(3, 2, figsize=(10, 12))

                    # Originaux
                    axs[0, 0].imshow(orig_img, cmap='gray')
                    axs[0, 0].set_title('Image originale')
                    axs[0, 0].axis('off')
                    axs[0, 1].imshow(orig_gt_rgb)
                    axs[0, 1].set_title('Masque original')
                    axs[0, 1].axis('off')

                    # Transformations (rotation + contraste)
                    axs[1, 0].imshow(transformed_img_np, cmap='gray')
                    axs[1, 0].set_title('Rotation + Intensité')
                    axs[1, 0].axis('off')
                    axs[1, 1].imshow(transformed_gt_rgb)
                    axs[1, 1].set_title('Masque transformé')
                    axs[1, 1].axis('off')

                    # Flip horizontal
                    axs[2, 0].imshow(flipped_img_np, cmap='gray')
                    axs[2, 0].set_title('Flip horizontal')
                    axs[2, 0].axis('off')
                    axs[2, 1].imshow(flipped_gt_rgb)
                    axs[2, 1].set_title('Masque flippé')
                    axs[2, 1].axis('off')

                    plt.tight_layout()
                    plt.savefig(os.path.join(aug_viz_dir, f'medical_augmentation_example_{idx + 1}.png'))
                    plt.close()

                # Appliquer les transformations élastiques
                if data_aug_elastic:
                    elastic_transform_instance = ElasticTransform()

                    # Préparer les images pour la transformation
                    img_tensor = torch.from_numpy(orig_img).unsqueeze(0).float()
                    gt_channel_tensor = torch.from_numpy(np.argmax(orig_gt, axis=-1)).unsqueeze(0).float()

                    # Appliquer la transformation avec la même seed
                    seed = np.random.randint(0, 1000000)
                    np.random.seed(seed)
                    elastic_img = elastic_transform_instance(img_tensor)
                    np.random.seed(seed)
                    elastic_gt = elastic_transform_instance(gt_channel_tensor)

                    # Convertir pour l'affichage
                    elastic_img_np = elastic_img.squeeze(0).numpy()
                    elastic_gt_np = elastic_gt.squeeze(0).numpy().astype(int)
                    elastic_gt_rgb = convert_mask_to_rgb_image(elastic_gt_np)

                    # Créer la figure
                    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

                    # Originaux
                    axs[0, 0].imshow(orig_img, cmap='gray')
                    axs[0, 0].set_title('Image originale')
                    axs[0, 0].axis('off')
                    axs[0, 1].imshow(orig_gt_rgb)
                    axs[0, 1].set_title('Masque original')
                    axs[0, 1].axis('off')

                    # Déformation élastique
                    axs[1, 0].imshow(elastic_img_np, cmap='gray')
                    axs[1, 0].set_title('Déformation élastique')
                    axs[1, 0].axis('off')
                    axs[1, 1].imshow(elastic_gt_rgb)
                    axs[1, 1].set_title('Masque déformé')
                    axs[1, 1].axis('off')

                    plt.tight_layout()
                    plt.savefig(os.path.join(aug_viz_dir, f'elastic_augmentation_example_{idx + 1}.png'))
                    plt.close()

        print(f"Visualisations d'augmentation de données générées dans {aug_viz_dir}")
        if args.predict is None:
            # Si nous ne sommes pas en mode prediction, nous pouvons terminer ici
            print(
                "Exécuter avec --data_aug_medical et/ou --data_aug_elastic pour appliquer ces augmentations à l'entraînement")
            exit(0)

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
    else:
        # Afficher des informations sur l'utilisation du GPU
        if use_cuda:
            print(
                f"Entraînement de {model.__class__.__name__} sur {args.dataset} pour {args.num_epochs} époques utilisant GPU: {torch.cuda.get_device_name(0)}")
        else:
            print(
                f"Entraînement de {model.__class__.__name__} sur {args.dataset} pour {args.num_epochs} époques utilisant CPU")
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
        if isinstance(model, UNet):
            model.save()  # save the model's weights for prediction (see help for more details)
            model_trainer.plot_image_mask_prediction()
        model_trainer.plot_metrics()
