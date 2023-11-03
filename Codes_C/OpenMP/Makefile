# Compiler
CC = gcc

# Flags
CFLAGS = -fopenmp

# Fichiers sources et exécutables
SRC1 = V1_pixel/OMP_Code_Sequentiel.c
EXE1 = V1_pixel/OMP_Pixel
SRC2 = V2_pixel_min_max/OMP_Code_Sequentiel.c
EXE2 = V2_pixel_min_max/OMP

all: $(EXE1) $(EXE2)

$(EXE1): $(SRC1)
	$(CC) $(CFLAGS) $(SRC1) -o $(EXE1)

$(EXE2): $(SRC2)
	$(CC) $(CFLAGS) $(SRC2) -o $(EXE2)

clean:
	rm -f $(EXE1) $(EXE2)
