import subprocess
import matplotlib.pyplot as plt

from datetime import datetime   # Récupére la date pour créer des fichiers uniques


# Compiler le fichier C
# subprocess.run(["nvcc", "-o", "V2_CUDA", "V2/V2_CUDA.cu"])

# Les différentes valeurs de threadsPerBlock2 que vous souhaitez tester
threadsPerBlock2_values = range(1, 1000, 50)  # Ajustez la plage et le pas selon vos besoins

# Pour stocker les temps d'exécution pour chaque valeur de threadsPerBlock2
execution_times = []

for threads in threadsPerBlock2_values:
    # Exécuter le fichier binaire avec la valeur courante de threadsPerBlock2
    result = subprocess.run(["V2/V2_CUDA", "../../Ressources/image1.pgm", str(threads)], 
                            capture_output=True, text=True)

    # Extraire le temps d'exécution de la dernière ligne
    time_line = result.stdout.strip().split('\n')[-1]
    time = float(time_line.split()[0])  # convertir la partie numérique de la ligne en float
    execution_times.append(time)
    print(f"Threads: {threads}, Time: {time} seconds")

# Tracer le graphique
plt.plot(threadsPerBlock2_values, execution_times, marker='o', linestyle='-', color='b')
plt.xlabel('Threads Per Block')
plt.ylabel('Execution Time (seconds)')
plt.title('Execution Time vs Threads Per Block')
plt.grid(True)
#plt.show()

now = datetime.now()
# Formater la date et l'heure au format désiré
formatted_date = now.strftime("%Y-%m-%d-%H-%M-%S")
fileName = "../../Graphes/CUDA/"+formatted_date+".png"

plt.savefig(fileName)
