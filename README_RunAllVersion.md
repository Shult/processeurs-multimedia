# Script pour exéciter toutes les versions du programme C

### Fichier : runAllVersion.py

**Auteur** : Sylvain MESTRE & Antoine MAURAIS  
**Date** : 19/09/2023

## Description

`runAllVersion.py` est un script Python conçu pour exécuter plusieurs versions d'un programme C qui traite des images. Après avoir exécuté toutes les versions, il affiche un graphique en blocs montrant le temps moyen d'exécution pour chaque version.

## Prérequis

- Assurez-vous d'avoir Python 3.x installé.
- Assurez-vous d'avoir les dépendances nécessaires. Vous pouvez les installer à l'aide de pip :

```bash
pip install matplotlib
```


- Votre programme C doit afficher le temps d'exécution sur la dernière ligne de sortie.

## Utilisation

1. Modifiez la variable `image_name` dans le script `runAllVersion.py` pour pointer vers votre image.
2. Ajoutez ou retirez des versions de votre programme C dans la liste `versions` du script `runAllVersion.py` selon vos besoins.
3. Exécutez le script avec la commande :

```bash
python3 runAllVersion.py
```


4. Observez le graphique en blocs montrant le temps moyen d'exécution pour chaque version.

## Remarques

- Le script fait appel à un autre script Python nommé `script.py`, qui est responsable de l'exécution de chaque version du programme C, de la collecte des temps d'exécution et de la génération de graphiques pour chaque version.
- Les graphiques générés par `script.py` sont sauvegardés dans un dossier nommé `Graphes`.

