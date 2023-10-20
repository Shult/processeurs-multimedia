import argparse
import subprocess
import multiprocessing
import matplotlib.pyplot as plt

from datetime import datetime   # Récupére la date pour créer des fichiers uniques


# Créer un parseur d'arguments
parser = argparse.ArgumentParser(description="Exécutez un programme OpenMP avec un nombre variable de threads et tracez les résultats.")

# Ajouter des arguments pour le chemin de l'exécutable et le chemin de l'image
parser.add_argument("executable", help="Le chemin vers le fichier exécutable OpenMP.")
parser.add_argument("image", help="Le chemin vers le fichier image à traiter.")
args = parser.parse_args()

# Obtenez le nombre de cœurs de la machine
num_cores = multiprocessing.cpu_count()
print(f"Number of cores : {num_cores}")

# Combien de fois chaque configuration de thread doit être exécutée
num_repeats = 80

# Listes pour stocker les résultats pour le graphique
threads = []
average_times = []

i = 0

for num_threads in range(1, num_cores):
    total_time = 0.0
    
    for i in range(num_repeats):
        # Exécutez le programme C avec le nombre spécifié de threads et collectez la sortie
        result = subprocess.run([args.executable, args.image, str(num_threads)], capture_output=True, text=True)

        # Extraire le temps d'exécution de la sortie
        # print(f"Result C : {result.stdout}")
        # time = float(result.stdout.split()[-2])  
        time = float(result.stdout)  

        total_time += time

    # Calculez la moyenne du temps d'exécution pour ce nombre de threads
    average_time = total_time / num_repeats
    print(f"Nombre de threads: {num_threads}, Temps d'exécution moyen: {average_time:.6f} s")

    # Ajouter les résultats aux listes pour le graphique
    threads.append(num_threads)
    average_times.append(average_time)

# Créer un graphique des résultats
plt.plot(threads, average_times, marker='o', linestyle='-', color='b')
plt.xlabel('Nombre de Threads')
plt.ylabel('Temps d\'Exécution Moyen (s)')
plt.title('Temps d\'Exécution Moyen par Nombre de Threads')
plt.grid(True)

# Pour sauvegarder le graphique
# Récupérer la date et l'heure courantes
now = datetime.now()
# Formater la date et l'heure au format désiré
formatted_date = now.strftime("%Y-%m-%d-%H-%M-%S")
# print(formatted_date)
fileName = "../../Graphes/OpenMP/"+formatted_date+".png"
plt.savefig(fileName)

# Afficher le graphique
plt.show()
