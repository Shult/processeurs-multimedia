import subprocess
from datetime import datetime
from matplotlib import pyplot as plt

def run_script_for_version(script_name, program_name, iterations, image_name, num_threads=None):
    # Exécuter le script Python pour une version spécifique du programme C
    command = ["python3", script_name, program_name, str(iterations), image_name]
    if num_threads is not None:
        command.append(str(num_threads))
    result = subprocess.run(command, capture_output=True, text=True)
    avg_time_line = [line for line in result.stdout.split('\n') if "Temps moyen d'exécution :" in line]
    if avg_time_line:
        avg_time = float(avg_time_line[0].split(":")[1].strip().split(" ")[0])
        return avg_time, result
    return None, result

if __name__ == "__main__":
    script_name = "script.py"
    iterations = 100
    image_name = "Ressources/image1.pgm"

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

    average_times = []
    for version, threads in versions.items():
        print(f"\nRunning for version: {version}\n")
        avg_time, result = run_script_for_version(script_name, version, iterations, image_name, threads)
        if avg_time is not None:
            average_times.append(avg_time)
        else:
            print(f"Erreur : le temps moyen n'a pas pu être calculé pour {version}")
            print("Sortie du script (stdout):\n", result.stdout)
            print("Sortie d'erreur du script (stderr):\n", result.stderr)
            average_times.append(0)


    labels = [version.split("/")[-1] for version in versions.keys()]

    plt.bar(range(len(versions)), average_times, tick_label=labels)
    plt.xlabel('Version')
    plt.ylabel('Temps moyen (us)')
    plt.title('Temps moyen d\'exécution par version')

    now = datetime.now()
    formatted_date = now.strftime("%Y-%m-%d-%H-%M-%S")
    fileName = "Graphes/GraphiquesBlocs/"+formatted_date+".png"
    plt.savefig(fileName)
    plt.show()
