# Makefile pour V4_AVX.c

CC = gcc
CFLAGS = -O3 -mavx2 -Wall

# Fichiers sources et exécutables
V1_src = V1/CodeSequentiel.c
V1_exe = V1/Init
V2_src = V2/CodeSequentiel_V2.c
V2_exe = V2/Vecteur
V3_src = V3_Float/CodeSequentiel_V3.c
V3_exe = V3_Float/Float
V4_src = V4_Short/CodeSequentiel_V4.c
V4_exe = V4_Short/Short
V5_src = V5_Char/CodeSequentiel_V5.c
V5_exe = V5_Char/Char

all: $(V1_exe) $(V2_exe) $(V3_exe) $(V4_exe) $(V5_exe)

$(V1_exe): $(V1_src)
	$(CC) $(CFLAGS) -o $(V1_exe) $(V1_src) -lm

$(V2_exe): $(V2_src)
	$(CC) $(CFLAGS) -o $(V2_exe) $(V2_src) -lm

$(V3_exe): $(V3_src)
	$(CC) $(CFLAGS) -o $(V3_exe) $(V3_src) -lm

$(V4_exe): $(V4_src)
	$(CC) $(CFLAGS) -o $(V4_exe) $(V4_src) -lm

$(V5_exe): $(V5_src)
	$(CC) $(CFLAGS) -o $(V5_exe) $(V5_src) -lm

clean:
	rm -f $(V1_exe) $(V2_exe) $(V3_exe) $(V4_exe) $(V5_exe)
