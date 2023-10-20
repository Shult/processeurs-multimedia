import subprocess

from datetime import datetime   # Récupére la date pour créer des fichiers uniques

from matplotlib import pyplot as plt

def run_script_for_version(script_name, program_name, iterations, image_name):
    # Exécuter le script Python pour une version spécifique du programme C
    result = subprocess.run(["python3", script_name, program_name, str(iterations), image_name], capture_output=True, text=True)
    avg_time_line = [line for line in result.stdout.split('\n') if "Temps moyen d'exécution :" in line]
    if avg_time_line:
        avg_time = float(avg_time_line[0].split(":")[1].strip().split(" ")[0])
        return avg_time
    return None

if __name__ == "__main__":
    script_name = "script2.py" # Fichier python qui execute un certain nombre de fois les programmes C.
    iterations = 100  # Nombre d'iterations pour chaque programme
    image_name = "Ressources/image1.pgm" # Image à traiter pour chaque programme

    # Liste de toutes vos versions de programmes C
    versions = [
        "./Codes_C/Code_Sequentiel/V1/a.out", 
        "./Codes_C/Code_Sequentiel/V2/a.out", 
        "./Codes_C/Code_Sequentiel/V3_Float/a.out", 
        "./Codes_C/Code_Sequentiel/V4_Short/a.out", 
        "./Codes_C/Code_Sequentiel/V5_Char/a.out",
        "./Codes_C/AVX/V4/a.out",
        "./Codes_C/CUDA/V1/CUDA_V1",
        "./Codes_C/CUDA/V1/CUDA_V2",
        "./Codes_C/OpenMP/V1_pixel/OMP_Code_Sequentiel",
        "./Codes_C/OpenMP/V2_pixel_min_max/OMP_Code_Sequentiel"
        ]
    average_times = []
    for version in versions:
        print(f"\nRunning for version: {version}\n")
        avg_time = run_script_for_version(script_name, version, iterations, image_name)
        average_times.append(avg_time)

    # Créer un graphique en blocs
    plt.bar(range(len(versions)), average_times, tick_label=[f"Version {i}" for i in range(1, len(versions) + 1)])
    plt.xlabel('Version')
    plt.ylabel('Temps moyen (us)')
    plt.title('Temps moyen d\'exécution par version')

    # Pour sauvegarder le graphique
    # Récupérer la date et l'heure courantes
    now = datetime.now()
    # Formater la date et l'heure au format désiré
    formatted_date = now.strftime("%Y-%m-%d-%H-%M-%S")
    # print(formatted_date)
    fileName = "Graphes/GraphiquesBlocs/"+formatted_date+".png"
    plt.savefig(fileName)

    plt.show()
