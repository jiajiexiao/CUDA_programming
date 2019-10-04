FILES = PiMC.cu
CC = nvcc
OUT_EXE = pimc

build: $(FILES)
	$(CC) -o $(OUT_EXE) $(FILES)

clean:
	rm -f $(OUT_EXE) core

