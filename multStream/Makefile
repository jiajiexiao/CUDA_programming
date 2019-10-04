FILES = PiHist.cu
CC = nvcc
OUT_EXE = pihist

build: $(FILES)
#	$(CC) -o $(OUT_EXE) $(FILES) -arch=sm_20 
	$(CC) -o $(OUT_EXE) $(FILES)
clean:
	rm -f $(OUT_EXE) core

