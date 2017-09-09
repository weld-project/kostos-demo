CC=gcc
FLAGS=-O3 -march=native -ffast-math -ftree-vectorize -std=c99

.PHONY: all asm clean

all: bench measure

bench:
	$(CC) $(FLAGS) bench.c -lm -o bench

measure:
	$(CC) $(FLAGS) measure.c -lm -o measure

asm:
	$(CC) $(FLAGS) bench.c -S -emit-llvm

clean:
	rm -rf bench measure bench.ll
