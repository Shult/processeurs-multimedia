# ==================================================================================================
# Programme : script.py
# Auteur : Sylvain MESTRE & Antoine MAURAIS
# Date : 19/09/2023
# Pré-requis : Le programme C doit afficher le temps d'exécution sur la dernière ligne du programme. 
# ==================================================================================================

import subprocess
import sys  
import matplotlib.pyplot as plt
from datetime import datetime

def run_program(program, iterations, image_name, num_threads=None):
    compiled_program = "./" + program
    execution_times = []

    for i in range(iterations):
        command = [compiled_program, image_name]
        if num_threads:
            command.append(num_threads)
        result = subprocess.run(command, capture_output=True, text=True)

        time_output = float(result.stdout.strip().split("\n")[-1].replace(" s", ""))
        execution_times.append(time_output)
        display_progress_bar(i, iterations)

    return execution_times

def plot_execution_times(execution_times):
    avg_time = sum(execution_times) / len(execution_times)
    plt.figure(figsize=(10,5))
    plt.plot(range(1, len(execution_times) + 1), execution_times, label='Temps d\'exécution', marker='o')

    plt.axhline(y=avg_time, color='red', linestyle='-', label=f'Temps moyen : {avg_time * 1000000:.2f} us')
    plt.xlabel('Numéro d\'exécution')
    plt.ylabel('Temps d\'exécution (s)')
    plt.title('Temps d\'exécution pour l\'image')
    plt.legend()
    plt.grid(True)

    now = datetime.now()
    formatted_date = now.strftime("%Y-%m-%d-%H-%M-%S")
    fileName = "Graphes/" + formatted_date + ".png"
    plt.savefig(fileName)
    plt.show()
    return avg_time

def display_progress_bar(iteration, total, bar_length=50):
    progress = (iteration + 1) / total
    arrow = '#' * int(round(progress * bar_length) - 1)
    spaces = ' ' * (bar_length - len(arrow))

    sys.stdout.write('\r[{0}] {1}%'.format(arrow + spaces, int(round(progress * 100))))
    sys.stdout.flush()

    if iteration == total - 1:
        print()

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python run_program.py [program_name] [number_of_executions] [image_name] [optional: number_of_threads]")
        sys.exit(1)

    program = sys.argv[1]
    iterations = int(sys.argv[2])
    image_name = sys.argv[3]

    num_threads = None
    if len(sys.argv) == 5:
        num_threads = sys.argv[4]

    execution_times = run_program(program, iterations, image_name, num_threads)
    avg_time = plot_execution_times(execution_times)
    print(f"Temps moyen d'exécution : {avg_time * 1000000:.2f} us")

    sys.exit(avg_time)
