# Comparaison des Temps d'Exécution GPU et CPU pour des Tailles de Vecteur Croissantes

### Fichier : scriptCUDA.py

**Auteur** : Sylvain MESTRE & Antoine MAURAIS  
**Date** : 19/09/2023

## Description

`scriptCUDA.py` est un script Python conçu pour comparer les performances d'un programme CUDA traitant des vecteurs de différentes tailles. Le script exécute le programme CUDA pour plusieurs tailles de vecteurs, mesure les temps d'exécution du GPU et du CPU, puis génère un graphique comparant ces temps.

## Prérequis

- Python 3.x
- Modules Python : `matplotlib`, `subprocess`, `re`. Installez-les avec la commande :

```bash
pip install matplotlib
```

- Le programme C doit afficher les temps d'exécution pour le GPU et le CPU sur les dernières lignes de sa sortie.

## Fonctionnement

Le script initie un vecteur de taille 10 et teste ensuite les performances du programme CUDA sur des vecteurs de taille croissante. Pour chaque taille de vecteur :

1. Le programme CUDA est exécuté plusieurs fois (défini par nb_iterations).
2. Les temps d'exécution du GPU et du CPU sont extraits de la sortie du programme et moyennés.
3. Ces moyennes sont stockées pour chaque taille de vecteur.
4. Les temps moyens d'exécution du GPU et du CPU pour toutes les tailles de vecteurs sont visualisés sur un graphique.

## Utilisation

1. Assurez-vous que le script a les permissions d'exécution :

```bash
chmod +x scriptCUDA.py
```

2. Exécutez le script :

```bash
python3 scriptCUDA.py
```

3. Le graphique comparant les temps d'exécution du GPU et du CPU pour différentes tailles de vecteurs sera sauvegardé dans le dossier ../../Graphes/CUDA/ avec un timestamp unique.

## Remarques

- L'échelle des axes x et y du graphique est logarithmique, offrant une meilleure visualisation pour une large plage de tailles de vecteurs.
- Assurez-vous que le programme C affiche correctement les temps d'exécution pour que ce script fonctionne correctement.