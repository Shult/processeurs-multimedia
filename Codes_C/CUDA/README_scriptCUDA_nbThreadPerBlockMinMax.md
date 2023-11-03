# Analyse de Performance CUDA en fonction du nombre de Threads par Bloc

### Fichier : scriptCUDA_nbThreadPerBlockMinMax.py

**Auteur** : Sylvain MESTRE & Antoine MAURAIS  
**Date** : 19/09/2023

## Description

`scriptCUDA_nbThreadPerBlockMinMax.py` est un script Python qui mesure et visualise les performances d'un programme CUDA en fonction du nombre de threads par bloc. L'objectif est d'identifier le nombre optimal de threads par bloc pour maximiser les performances.

## Prérequis

- Python 3.x
- Module Python : `matplotlib`. Installez-le avec la commande :

```bash
pip install matplotlib
```

- Compiler le programme CUDA correspondant. Le script contient une commande commentée pour compiler avec nvcc, assurez-vous d'avoir nvcc installé.

## Fonctionnement

Le script teste une plage de valeurs pour le nombre de threads par bloc (défini par threadsPerBlock2_values). Pour chaque valeur :

1. Le programme CUDA est exécuté avec l'image ../../Ressources/image1.pgm.
2. Le temps d'exécution est extrait de la sortie du programme.
3. Les temps d'exécution pour toutes les valeurs testées sont ensuite visualisés sur un graphique.

## Utilisation

1. Assurez-vous que le script a les permissions d'exécution :

```bash
chmod +x scriptCUDA_nbThreadPerBlockMinMax.py
```

2. Exécutez le script :

```bash
python3 scriptCUDA_nbThreadPerBlockMinMax.py
```

3. Les résultats seront affichés dans la console et le graphique des temps d'exécution vs le nombre de threads par bloc sera sauvegardé dans le dossier Graphes/CUDA/ avec un timestamp unique.

Remarques

- Le script actuellement teste le nombre de threads par bloc de 1 à 32 par incréments de 1. Modifiez la plage threadsPerBlock2_values selon vos besoins.
- Le programme CUDA doit afficher le temps d'exécution sur la dernière ligne de sa sortie pour que ce script fonctionne correctement.

