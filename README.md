# Rapport

# IFT 780 - Devoir 4 - Segmentation d'images IRM cardiaques

Ce projet implémente des réseaux de neurones pour la segmentation d'images IRM cardiaques en quatre classes :
1. Fond
2. Ventricule droit
3. Myocarde
4. Ventricule gauche

## Fichiers modifiés

### Question 1 - Architectures

1. `src/models/YourUNet.py`
   - Nouvelle architecture basée sur UNet avec :
     - Convolutions séparables
     - Couches résiduelles
     - Blocs bottleneck

2. `src/models/YourSegNet.py`
   - Architecture non-symétrique inspirée de DeepLabV3 avec :
     - Encoder profond avec atrous convolutions
     - Decoder léger avec upsampling progressif
     - Skip connections asymétriques

3. `src/utils/losses.py`
   - Nouvelles fonctions de perte :
     - `DiceLoss` : Combine la perte Dice avec l'entropie croisée
     - `FocalLoss` : Gestion du déséquilibre des classes

4. `/train.py`
   - Ajout des nouveaux modèles dans les options
   - Intégration des nouvelles fonctions de perte
   - Support du dataset ACDC

### Question 2 - Checkpointing

1. `src/manage/CNNTrainTestManager.py`
   - Ajout du système de checkpointing
   - Sauvegarde automatique du meilleur modèle
   - Sauvegarde des checkpoints à chaque époque
   - Possibilité de charger un checkpoint existant

## Commandes d'exécution

### Entraînement des modèles

1. YourUNet avec Dice Loss :
```bash
python src/train.py --model YourUNet --loss Dice --num-epochs 50 --batch_size 16 --lr 0.0001
```

2. YourSegNet avec Focal Loss :
```bash
python src/train.py --model YourSegNet --loss Focal --num-epochs 50 --batch_size 16 --lr 0.0001
```

### Prédiction avec un modèle entraîné
```bash
python src/train.py --model YourUNet --predict path/to/weights.pth
```

### Checkpointing
Les checkpoints sont automatiquement sauvegardés dans le dossier `checkpoints/` :
- `checkpoint_epoch_X.pth` : Checkpoint de chaque époque
- `best_model.pth` : Meilleur modèle selon la précision de validation

Pour charger un checkpoint spécifique :
```bash
python train.py --model YourUNet --checkpoint checkpoints/best_model.pth
```

Pour continuer l'entraînement à partir d'un checkpoint :
```bash
python train.py --model YourUNet --resume checkpoints/checkpoint_epoch_10.pth
```

## Description des nouvelles fonctions de perte

### Dice Loss
La perte Dice combine la perte Dice avec l'entropie croisée pour une meilleure stabilité :
- Calcule le coefficient de Dice pour chaque classe
- Combine avec l'entropie croisée standard
- Utile pour la segmentation car elle prend en compte le chevauchement entre les prédictions et les cibles

### Focal Loss
La perte Focal est conçue pour gérer le déséquilibre des classes :
- Réduit automatiquement le poids des exemples faciles
- Augmente le poids des exemples difficiles
- Paramètres ajustables (alpha et gamma) pour contrôler le focus

## Courbes d'entraînement
[À remplir apres l'entraînement des model]

![fig1.png](figures/YourSegNet_metrics-dice-e10.png)