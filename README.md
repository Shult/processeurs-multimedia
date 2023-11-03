# Projet de Traitement d'Images en Parallèle

Ce projet vise à comparer différentes méthodes d'optimisation pour augmenter le contraste d'une image. Il utilise plusieurs techniques de parallélisation, y compris SSE, AVX, CUDA et OMP.

## Structure du Projet

- **runAllVersion.py** : Exécute toutes les versions du programme pour traiter une image et affiche un graphique en blocs montrant le temps moyen d'exécution pour chaque version. Consultez `README_RunAllVersion` pour plus d'informations.
- **script.py** : Exécute un programme C donné plusieurs fois, mesure et affiche le temps d'exécution pour chaque exécution. Consultez `README_Script.md` pour plus de détails.
- **Codes_C** : Contient les codes sources des différentes versions.
  - **AVX** : Version optimisée avec Advanced Vector Extensions.
  - **OMP** : Version optimisée avec OpenMP.
  - **Code_Sequentiel** : Version utilisant SSE.
  - **CUDA** : Version optimisée pour les GPUs avec CUDA.
- **Ressources** : Dossier contenant les images utilisées et traitées dans ce projet.
- **Graphes** : Résultats du projet sous forme de graphiques, y compris les graphiques générés par `script.py`.

## Prérequis

- Python 3.x
- Modules Python : `subprocess`, `sys`, `matplotlib`. Vous pouvez installer les dépendances avec :

```bash
pip install matplotlib
```


## Comment démarrer

1. **Exécution de toutes les versions** :
    - Modifiez la variable `image_name` dans `runAllVersion.py` pour pointer vers votre image. Par default on exécute "/Ressources/image1.pgm"
    - Ajoutez ou retirez des versions de votre programme C dans la liste `versions` du script selon vos besoins.
    
    Par exemple : 

    ```Python
    versions = {
        #"./Codes_C/Code_Sequentiel/V1/Init": None, 
        "./Codes_C/Code_Sequentiel/V2/Vecteur": None, 
        "./Codes_C/Code_Sequentiel/V3_Float/Float": None, 
        "./Codes_C/Code_Sequentiel/V4_Short/Short": None, 
        "./Codes_C/Code_Sequentiel/V5_Char/Char": None,
        "./Codes_C/AVX/V4/AVX": None,
        #"./Codes_C/CUDA/V1/V1_CUDA": None,
        "./Codes_C/CUDA/V2/V2_CUDA": 10, # Taille des blocs
        "./Codes_C/OpenMP/V2_pixel_min_max/OMP": 9 # Nombre de thread(s)
        # ... (ajouter tous les autres chemins de fichier exécutables ici)
    }
    ```

    - Exécutez le script :
      ```
      python3 runAllVersion.py
      ```

2. **Exécution individuelle des programmes avec mesure des performances** :
    - Exécutez `script.py` en précisant le programme C à exécuter, le nombre d'exécutions, et le nom de l'image :
      ```
      python script.py [nom_programme_C][nombre_exécutions] [nom_image]
      ```
      Exemple :
      ```
      python script.py Codes_C/CodeSequentiel/V1/Init 5 Ressources/image.jpg
      ```

3. **Compilation des sources** :
    Dans chaque sous-dossier de `Codes_C`, utilisez :

    ```bash 
    make clean
    make all
    ```
    Par exemple, pour compiler la version AVX :

    ```bash 
    cd Codes_C/CUDA
    make clean
    make all
    ```

## Ressources

Les images pour ce projet se trouvent dans le dossier `Ressources`. Les graphiques des temps d'exécution sont sauvegardés dans `Graphes`.

**Documentation utile** : [Documentation intel : Intrinsics Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#)

## Auteurs

- [Sylvain MESTRE](https://www.linkedin.com/in/sylvain-mestre-22173a190/)
- Antoine MAURAIS

## Besoin d'aide ?

Si vous avez des questions ou si vous rencontrez des problèmes lors de l'utilisation de ce projet, n'hésitez pas à me contacter à mestres.sin@gmail.com
