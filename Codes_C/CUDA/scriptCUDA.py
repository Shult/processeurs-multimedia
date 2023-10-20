# ==================================================================================================
# Programme : script.py
# Auteur : Sylvain MESTRE & Antoine MAURAIS
# Date : 19/09/2023
# Pré-requis : Le programme C doit affiché le temps d'exécution sur la dernière ligne du programme. 
# ==================================================================================================

import subprocess
import matplotlib.pyplot as plt
import re

from datetime import datetime   # Récupére la date pour créer des fichiers uniques


# Taille initiale du vecteur
taille_initiale = 10

# Stocker les temps d'exécution GPU et CPU
temps_gpu = []
temps_cpu = []
tailles = []

# Nombre d'itérations pour chaque nombre de threads par bloc
nb_iterations = 5  # Vous pouvez ajuster cette valeur selon vos besoins

# Exécuter le programme CUDA avec des tailles de vecteur croissantes
for i in range(1, 7):  # Modifier la plage selon vos besoins
    taille = taille_initiale * (10 ** i)
    tailles.append(taille)

    temps_gpu_total = 0
    temps_cpu_total = 0
    
    for _ in range(nb_iterations):
        # Exécuter le programme CUDA et récupérer la sortie
        resultat = subprocess.run(["./Exemple/VecteurGPU", str(taille), "10"], capture_output=True, text=True)
    
        # Extraire les temps d'exécution GPU et CPU à partir de la sortie
        sortie = resultat.stdout
        gpu_time = int(re.search(r"Temps total pour le GPU = (\d+)", sortie).group(1))
        cpu_time = int(re.search(r"Temps total CPU : (\d+)", sortie).group(1))

        temps_gpu_total += gpu_time
        temps_cpu_total += cpu_time
        
    temps_gpu.append(gpu_time)
    temps_cpu.append(cpu_time)

# Dessiner le graphique
plt.figure(figsize=(10, 5))
plt.plot(tailles, temps_gpu, marker='o', label='GPU', color='b')
plt.plot(tailles, temps_cpu, marker='x', label='CPU', color='r')
plt.xscale('log')
plt.yscale('log')  # Ajoutez cette ligne pour une échelle logarithmique sur l'axe des y
plt.xlabel('Taille du vecteur')
plt.ylabel('Temps d\'exécution (microsecondes)')
plt.title('Comparaison des temps d\'exécution GPU et CPU')
plt.legend()
plt.grid(True)

now = datetime.now()
# Formater la date et l'heure au format désiré
formatted_date = now.strftime("%Y-%m-%d-%H-%M-%S")
fileName = "../../Graphes/CUDA/"+formatted_date+".png"

plt.savefig(fileName)
#plt.show()
