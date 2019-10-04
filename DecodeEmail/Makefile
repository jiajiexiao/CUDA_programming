FILES = main.cu
CC = nvcc
OUT_EXE = decode

build: $(FILES)
	$(CC) -o $(OUT_EXE) $(FILES)

clean:
	rm -f $(OUT_EXE) core

