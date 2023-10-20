# Script Python pour l'Exécution Parallèle OpenMP

## Description
Ce script Python est conçu pour exécuter un programme parallèle OpenMP avec un nombre variable de threads et pour tracer les résultats. Il exécute le programme spécifié avec un nombre de threads allant de 1 au nombre total de cœurs CPU disponibles sur la machine. Chaque configuration de threads est exécutée plusieurs fois pour calculer le temps d'exécution moyen, et les résultats sont ensuite tracés dans un graphique pour une analyse visuelle.

## Dépendances
- **Python** (testé avec la version 3.10)
- **Matplotlib** pour le traçage des graphiques
- Un programme **OpenMP** compilé pour être testé

Installez Matplotlib avec pip :
```sh
pip install matplotlib
```

## Utilisation
Pour utiliser le script, exécutez-le avec Python, en passant le chemin vers l'exécutable OpenMP et le chemin vers l'image comme arguments. Par exemple :

```sh
python script_nbr_thread_omp.py ./V1/OMP_Code_Sequentiel ../../Ressources/image1.pgm
```

Le script exécutera le programme OpenMP avec un nombre de threads allant de 1 au nombre total de cœurs CPU sur la machine. Chaque configuration est exécutée un certain nombre de fois pour calculer un temps d'exécution moyen. Les résultats seront ensuite tracés dans un graphique.

## Paramètres

Le script prend deux arguments obligatoires :

1. **executable** : Le chemin vers l'exécutable OpenMP.
2. **image** : Le chemin vers le fichier image à traiter avec le programme OpenMP.

## Sortie

Le script imprime le temps d'exécution moyen pour chaque configuration de threads dans la console et affiche un graphique des résultats à la fin de l'exécution. Le graphique montre le temps d'exécution moyen par rapport au nombre de threads, permettant une analyse visuelle des performances du programme parallèle.