# Script d'exécution et de mesure de performance

### Fichier : script.py

## Description
Ce script exécute un programme C donné un certain nombre de fois, mesure le temps d'exécution de chaque run, et affiche ensuite un graphique montrant les temps d'exécution ainsi que le temps moyen.

## Auteurs
- Sylvain MESTRE
- Antoine MAURAIS

## Date
19/09/2023

## Pré-requis

- Assurez-vous que le programme C affiche le temps d'exécution sur la dernière ligne de son output.
- Vous aurez besoin de Python 3.x et des modules suivants :
  - `subprocess`
  - `sys`
  - `matplotlib`

## Utilisation

### Exécution du script

Pour exécuter le script :

```bash
python script.py [nom_programme_C][nombre_exécutions] [nom_image]
```

### Exemple d'utilisation :

```bash
python script.py CodeSequentiel 5 image.jpg
```

## Fonctionnalités

- **Exécution du programme C** : Le programme C spécifié est exécuté un nombre donné de fois.
- **Mesure des temps d'exécution** : Après chaque exécution, le temps d'exécution est extrait depuis la sortie du programme C et stocké.
- **Affichage des temps d'exécution** : Tous les temps d'exécution mesurés sont affichés à la console.
- **Visualisation graphique** : Un graphique est généré montrant tous les temps d'exécution, ainsi que le temps moyen d'exécution.

## Ressources

- [Documentation intel : Intrinsics Guide ](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#)