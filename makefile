CC = gcc -lpthread -lm

all: original cpu

original : Original_implementation.c
	$(CC) Original_implementation.c -o original 

cpu : CPU_parallelized.c
	$(CC) CPU_parallelized.c -o cpu 


clean:
	rm original cpu