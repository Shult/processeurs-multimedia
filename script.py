# ==================================================================================================
# Programme : script.py
# Auteur : Sylvain MESTRE & Antoine MAURAIS
# Date : 19/09/2023
# Pré-requis : Le programme C doit affiché le temps d'exécution sur la dernière ligne du programme. 
# ==================================================================================================

import subprocess # Permet d'exécuter des commandes système à partir du script Python
import sys  # Fournit un accès aux variables et fonctions liées au système.
import matplotlib.pyplot as plt # Bibliothèque pour créer des graphique et des tracés. 

def run_program(program, iterations, image_name):
    compiled_program = "./"+program
    
    # Liste pour stocker les temps d'exécution
    execution_times = []

    # Boucle pour executer n fois le programme passé en paramètre
    for i in range(iterations):
        result = subprocess.run([compiled_program, image_name], capture_output=True, text=True)
        
        # Récupérer le temps d'exécution depuis la sortie du programme, qui est à la dernière ligne de sortie du programme
        time_output = float(result.stdout.strip().split("\n")[-1].replace(" s", ""))
        execution_times.append(time_output)

        # Mise à jour de la barre de chargement
        display_progress_bar(i, iterations)
    
    # Afficher les temps d'exécution
    for index, time in enumerate(execution_times, 1):
        print(f"Exécution {index}: {time}")

    # Return the times for plotting
    return execution_times

# Affiche un graphique des temps d'exécution 
def plot_execution_times(execution_times):
    avg_time = sum(execution_times) / len(execution_times)
    print("Temps moyen d'exécution : ", avg_time*1000000, " ms")
    plt.figure(figsize=(10,5))
    plt.plot(range(1, len(execution_times) + 1), execution_times, label='Temps d\'exécution', marker='o')

    plt.axhline(y=avg_time, color='red', linestyle='-', label=f'Temps moyen : {avg_time * 1000000:.2f} ms')
    plt.xlabel('Numéro d\'exécution')
    plt.ylabel('Temps d\'exécution (s)')
    plt.title('Temps d\'exécution pour l\'image')
    plt.legend()
    plt.grid(True)
    plt.show()

# Affiche une barre de progression
def display_progress_bar(iteration, total, bar_length=50):
    progress = (iteration + 1) / total
    arrow = '#' * int(round(progress * bar_length) - 1)
    spaces = ' ' * (bar_length - len(arrow))

    sys.stdout.write('\r[{0}] {1}%'.format(arrow + spaces, int(round(progress * 100))))
    sys.stdout.flush()

    if iteration == total - 1:
        print() # Retour à la ligne


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python run_program.py [number_of_executions] [image_name]")
        sys.exit(1)

    program = sys.argv[1]   # Récupère le premier paramètre
    iterations = int(sys.argv[2])   # Récupère le deuxième paramètre
    image_name = sys.argv[3]    # Récupère le troisième paramètre
    
    execution_times = run_program(program, iterations, image_name)

    # Graphique des temps d'exécution
    plot_execution_times(execution_times)
