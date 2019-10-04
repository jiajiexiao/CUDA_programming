FILES = PiHist.cu
CC = nvcc
OUT_EXE = pihist

build: $(FILES)
	$(CC) -o $(OUT_EXE) $(FILES) -arch=sm_20

clean:
	rm -f $(OUT_EXE) core

