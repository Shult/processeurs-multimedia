# Chemin vers le compilateur NVCC de CUDA
NVCC = nvcc

# Options de compilation, ajoutez des optimisations ou d'autres options au besoin
NVCCFLAGS = -O3 -arch=sm_35

# Tous les exécutables à construire
all: V1/V1_CUDA V2/V2_CUDA

# Règle pour construire le premier exécutable
V1/V1_CUDA: V1/V1_CUDA.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@

# Règle pour construire le deuxième exécutable
V2/V2_CUDA: V2/V2_CUDA.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@

# Règle pour nettoyer les exécutables et les objets
clean:
	rm -f V1/V1_CUDA V2/V2_CUDA
