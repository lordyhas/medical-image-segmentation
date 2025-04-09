# -*- coding:utf-8 -*-

"""
University of Sherbrooke
Date:
Authors: Mamadou Mountagha BAH & Pierre-Marc Jodoin
License:
Other: Suggestions are welcome
"""

import warnings
import torch
import numpy as np
import os
from src.manage.DataManager import DataManager
from typing import Callable, Type
from tqdm import tqdm
from src.models.UNet import UNet
from src.models.yourUNet import YourUNet
from src.models.yourSegNet import YourSegNet
from src.utils.utils import mean_dice, convert_mask_to_rgb_image
import matplotlib.pyplot as plt


class CNNTrainTestManager(object):
    """
    Class used the train and test the given model in the parameters
    """

    def __init__(self, model,
                 trainset: torch.utils.data.Dataset,
                 testset: torch.utils.data.Dataset,
                 loss_fn: torch.nn.Module,
                 optimizer_factory: Callable[[torch.nn.Module], torch.optim.Optimizer],
                 batch_size=1,
                 validation=None,
                 use_cuda=False,
                 checkpoint_dir='checkpoints'):
        """
        Args:
            model: model to train
            trainset: dataset used to train the model
            testset: dataset used to test the model
            loss_fn: the loss function used
            optimizer_factory: A callable to create the optimizer. see optimizer function
            below for more details
            validation: wether to use custom validation data or let the one by default
            use_cuda: to Use the gpu to train the model
            checkpoint_dir: directory to save model checkpoints
        """
        device_name = 'cuda:0' if use_cuda else 'cpu'
        if use_cuda and not torch.cuda.is_available():
            warnings.warn("CUDA is not available. Suppress this warning by passing "
                          "use_cuda=False to {}()."
                          .format(self.__class__.__name__), RuntimeWarning)
            device_name = 'cpu'

        self.device = torch.device(device_name)
        if validation is not None:
            self.data_manager = DataManager(trainset, testset, batch_size=batch_size, validation=validation)
        else:
            self.data_manager = DataManager(trainset, testset, batch_size=batch_size)
        self.loss_fn = loss_fn
        self.model = model

        # Transférer le modèle et la fonction de perte vers le GPU si disponible
        self.model = self.model.to(self.device)
        if hasattr(self.loss_fn, 'to'):
            self.loss_fn = self.loss_fn.to(self.device)

        self.optimizer = optimizer_factory(self.model)
        self.use_cuda = use_cuda
        self.metric_values = {}

        # Initialisation du checkpointing
        self.checkpoint_dir = checkpoint_dir
        self.best_val_acc = 0.0
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Afficher la taille du modèle (nombre de paramètres)
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total des paramètres: {total_params:,}")
        print(f"Paramètres entraînables: {trainable_params:,}")

    def save_checkpoint(self, epoch, is_best=False):
        """
        Sauvegarde un checkpoint du modèle
        Args:
            epoch: numéro de l'époque actuelle
            is_best: si True, sauvegarde comme meilleur modèle
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_acc': self.metric_values['val_acc'][-1],
            'train_acc': self.metric_values['train_acc'][-1],
            'val_loss': self.metric_values['val_loss'][-1],
            'train_loss': self.metric_values['train_loss'][-1],
            'device': str(self.device)
        }

        model_name = self.model.__class__.__name__

        # Sauvegarde du checkpoint de l'époque
        torch.save(checkpoint, os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}_{model_name}.pth'))

        # Si c'est le meilleur modèle, on le sauvegarde séparément
        if is_best:
            torch.save(checkpoint, os.path.join(self.checkpoint_dir, f'best_model_{model_name}.pth'))
            print(
                f'Nouveau meilleur modèle sauvegardé avec une précision de validation de {self.metric_values["val_acc"][-1]:.3f}')

    def load_checkpoint(self, checkpoint_path):
        """
        Charge un checkpoint du modèle
        Args:
            checkpoint_path: chemin vers le fichier de checkpoint
        """
        # Assurer le chargement sur le bon dispositif
        map_location = self.device
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Si le checkpoint a été sauvegardé sur GPU mais chargé sur CPU, ajuster les états de l'optimiseur
        if 'device' in checkpoint and checkpoint['device'] != str(self.device):
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)

        print(f"Checkpoint chargé depuis l'époque {checkpoint['epoch']}")
        return checkpoint['epoch']

    def evaluate_on_validation_set(self):
        """
        function that evaluate the model on the validation set every epoch
        """
        # switch to eval mode so that layers like batchnorm's layers nor dropout's layers
        # works in eval mode instead of training mode
        self.model.eval()

        # Get validation data
        val_loader = self.data_manager.get_validation_set()
        validation_loss = 0.0
        validation_losses = []
        validation_accuracies = []

        with torch.no_grad():
            for j, val_data in enumerate(val_loader, 0):
                # transfer tensors to the selected device
                val_inputs, val_labels = val_data[0].to(self.device), val_data[1].to(self.device)

                # forward pass
                val_outputs = self.model(val_inputs)

                # compute loss function
                loss = self.loss_fn(val_outputs, val_labels)
                validation_losses.append(loss.item())
                validation_accuracies.append(self.accuracy(val_outputs, val_labels))
                validation_loss += loss.item()

        self.metric_values['val_loss'].append(np.mean(validation_losses))
        self.metric_values['val_acc'].append(np.mean(validation_accuracies))

        # Vérifie si c'est le meilleur modèle
        current_val_acc = np.mean(validation_accuracies)
        is_best = current_val_acc > self.best_val_acc
        if is_best:
            self.best_val_acc = current_val_acc

        # displays metrics
        print('Validation loss %.3f' % (validation_loss / len(val_loader)))
        print('Validation accuracy %.3f' % (np.mean(validation_accuracies)))

        # switch back to train mode
        self.model.train()

        return is_best

    def train(self, num_epochs):
        """
        Train the model for num_epochs times
        Args:
            num_epochs: number times to train the model
        """
        # Initialize metrics container
        self.metric_values['train_loss'] = []
        self.metric_values['train_acc'] = []
        self.metric_values['val_loss'] = []
        self.metric_values['val_acc'] = []

        # Create pytorch's train data_loader
        train_loader = self.data_manager.get_train_set()
        # train num_epochs times
        for epoch in range(num_epochs):
            print("Epoch: {} of {}".format(epoch + 1, num_epochs))
            train_loss = 0.0
            train_acc = 0.0

            with tqdm(range(len(train_loader))) as t:
                train_losses = []
                train_accuracies = []
                for i, data in enumerate(train_loader, 0):
                    # transfer tensors to selected device
                    train_inputs, train_labels = data[0].to(self.device), data[1].to(self.device)

                    # zero the parameter gradients
                    self.optimizer.zero_grad()

                    # forward pass
                    train_outputs = self.model(train_inputs)
                    # computes loss using loss function loss_fn
                    loss = self.loss_fn(train_outputs, train_labels)

                    # Use autograd to compute the backward pass.
                    loss.backward()

                    # updates the weights using gradient descent
                    """
                    Way it could be done manually

                    with torch.no_grad():
                        for param in self.model.parameters():
                            param -= learning_rate * param.grad
                    """
                    self.optimizer.step()

                    # Save losses for plotting purposes
                    train_losses.append(loss.item())
                    acc = self.accuracy(train_outputs, train_labels)
                    train_accuracies.append(acc)

                    # print metrics along progress bar
                    train_loss += loss.item()
                    train_acc += acc
                    # t.set_postfix({loss='{:05.3f}'.format(train_loss / (i + 1))})
                    t.set_postfix({'Train loss': train_loss / (i + 1), "train_acc": train_acc / (i + 1)})
                    t.update()

            # evaluate the model on validation data after each epoch
            self.metric_values['train_loss'].append(np.mean(train_losses))
            self.metric_values['train_acc'].append(np.mean(train_accuracies))
            is_best = self.evaluate_on_validation_set()

            # Sauvegarde du checkpoint
            self.save_checkpoint(epoch, is_best)

        print("Finished training.")

    def accuracy(self, outputs, labels):
        """
        Computes the accuracy of the model
        Args:
            outputs: outputs predicted by the model
            labels: real outputs of the data
        Returns:
            Accuracy of the model
        """
        if isinstance(self.model, UNet) or isinstance(self.model, YourSegNet) or isinstance(self.model, YourUNet):
            # compute the mean of the 3 classes's dice score
            return mean_dice(outputs, labels).item()
        else:
            predicted = outputs.argmax(dim=1)
            correct = (predicted == labels).sum().item()
            return correct / labels.size(0)

    def evaluate_on_test_set(self):
        """
        Evaluate the model on the test set
        :returns;
            Accuracy of the model on the test set
        """
        test_loader = self.data_manager.get_test_set()
        accuracies = 0
        with torch.no_grad():
            for data in test_loader:
                test_inputs, test_labels = data[0].to(self.device), data[1].to(self.device)
                test_outputs = self.model(test_inputs)
                accuracies += self.accuracy(test_outputs, test_labels)
        print("Accuracy of the network on the test set: {:05.3f} %".format(100 * accuracies / len(test_loader)))

    def plot_metrics(self):
        """
        Function that plots train and validation losses and accuracies after training phase
        """
        epochs = range(1, len(self.metric_values['train_loss']) + 1)

        f = plt.figure(figsize=(10, 5))
        ax1 = f.add_subplot(121)
        ax2 = f.add_subplot(122)

        # loss plot
        ax1.plot(epochs, self.metric_values['train_loss'], '-o', label='Training loss')
        ax1.plot(epochs, self.metric_values['val_loss'], '-o', label='Validation loss')
        ax1.set_title('Training and validation loss')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.legend()

        # accuracy plot
        ax2.plot(epochs, self.metric_values['train_acc'], '-o', label='Training accuracy')
        ax2.plot(epochs, self.metric_values['val_acc'], '-o', label='Validation accuracy')
        ax2.set_title('Training and validation accuracy')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('accuracy')
        ax2.legend()
        # f.savefig('fig1.png')

        # Save figure with model-specific name
        model_name = self.model.__class__.__name__  # os.path.basename(self.checkpoint_dir)
        figures_dir = 'figures'
        os.makedirs(figures_dir, exist_ok=True)
        fig_path = os.path.join(figures_dir, f'{model_name}_metrics.png')
        f.savefig(fig_path)
        print(f'Métriques sauvegardées dans: {fig_path}')

        plt.show()

    def plot_image_mask_prediction(self):
        """
        Function that plots an image its corresponding ground truth and the predicted mask
        """

        for it in range(2, 5):
            # pick randomly an image and its corresponding gt into test_set
            img, gt = self.data_manager.get_random_sample_from_test_set()
            # convert the ground truth to a rgb image
            gt = convert_mask_to_rgb_image(gt)
            # use the model to predict the segmented image
            # Since the model expect a 4D we need to add the batch dim in order to get the 4D
            img = np.expand_dims(img, axis=0)
            # convert image to Tensor
            img = torch.from_numpy(img)
            img = img.to(self.device)
            prediction = self.model(img)
            # delete the batch dimension
            prediction = np.squeeze(prediction)
            # convert prediction to numpy array
            # take into account if model is trained on cpu or gpu
            if self.use_cuda and torch.cuda.is_available():
                prediction = prediction.detach().cpu().numpy()
            else:
                prediction = prediction.detach().numpy()
            # from one_hot vector to categorical
            prediction = np.argmax(prediction, axis=0)
            # convert the predicted mask to rgb image
            prediction = convert_mask_to_rgb_image(prediction)
            # remove the batch dim and the channel dim of img
            img = np.squeeze(np.squeeze(img))
            # convert img to a numpy array
            if self.use_cuda and torch.cuda.is_available():
                img = img.cpu().numpy()
            else:
                img = img.numpy()

            # plot the image
            f = plt.figure(figsize=(10, 10))
            ax1 = f.add_subplot(221)
            ax1.imshow(img)
            ax1.set_title('image')
            ax1.axis('off')

            # plot the ground truth
            ax2 = f.add_subplot(222)
            ax2.imshow(gt.astype('uint8'))
            ax2.axis('off')
            ax2.set_title('ground truth')

            # plot the image
            ax3 = f.add_subplot(223)
            ax3.imshow(img)
            ax3.set_title('image')
            ax3.axis('off')

            # plot the predicted mask
            ax4 = f.add_subplot(224)
            ax4.imshow(prediction.astype('uint8'))
            ax4.set_title('predicted segmentation')
            ax4.axis('off')
            # # Save as a png image
            # f.savefig('fig{}.png'.format(it+1))

            # Save as a png image with model-specific name
            model_name = self.model.__class__.__name__  # os.path.basename(self.checkpoint_dir)
            figures_dir = 'figures'
            os.makedirs(figures_dir, exist_ok=True)
            fig_path = os.path.join(figures_dir, f'{model_name}_prediction_{it}.png')
            f.savefig(fig_path)
            print(f'Prédiction sauvegardée dans: {fig_path}')

            # show image
            plt.show()


def optimizer_setup(optimizer_class: Type[torch.optim.Optimizer], **hyperparameters) -> \
        Callable[[torch.nn.Module], torch.optim.Optimizer]:
    """
    Creates a factory method that can instanciate optimizer_class with the given
    hyperparameters.

    Why this? torch.optim.Optimizer takes the model's parameters as an argument.
    Thus we cannot pass an Optimizer to the CNNBase constructor.

    Args:
        optimizer_class: optimizer used to train the model
        **hyperparameters: hyperparameters for the model
        Returns:
            function to setup the optimizer
    """

    def f(model):
        return optimizer_class(model.parameters(), **hyperparameters)

    return f
