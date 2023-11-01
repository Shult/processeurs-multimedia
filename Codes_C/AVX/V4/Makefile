# Makefile pour V4_AVX.c

CC = g++
CFLAGS = -O3 -mavx2 -Wall
TARGET = a.out
SRC = V4_AVX.c

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC) -lm

clean:
	rm -f $(TARGET)
