/*
CSC691 GPU programming
Project 1: All Hail the New Caesar!
Jiajie Xiao
Sep 14, 2017
*/

#include <stdio.h> 
#define CHUNK 1024 

__global__ void minus (char *cipherTxt, char *plainTxt)
{
        int idx = threadIdx.x;
        plainTxt[idx] = cipherTxt[idx]-1; // caesar cipher
}

void decode (char *cipherTxt, char *plainTxt)
{
        char *dev_cipherTxt, *dev_plainTxt;
        size_t size = CHUNK * sizeof(char);
        
        cudaMalloc((void **)&dev_cipherTxt, size);
        cudaMalloc((void **)&dev_plainTxt, size);
        cudaMemcpy(dev_cipherTxt, cipherTxt, size, cudaMemcpyHostToDevice);

        // start decoding the caesar cipher
        minus<<<1, CHUNK>>>(dev_cipherTxt, dev_plainTxt);
        cudaThreadSynchronize();
        cudaMemcpy(plainTxt, dev_plainTxt, size, cudaMemcpyDeviceToHost);

        cudaFree(dev_plainTxt);
        cudaFree(dev_cipherTxt);
}

void decode_serial (char *cipherTxt, char *plainTxt)
{
	for (int idx=0;idx<CHUNK*sizeof(char);idx++)	
                plainTxt[idx] = cipherTxt[idx] - 1; // caesar cipher
}

int main(int argc, char **argv)
{
        FILE *inputFile, *outputFile; 

        if (argc < 2)
        {
                printf("An input file name is required.");
                return -1;
        }
        else
                inputFile = fopen(argv[1],"r");

        if (ferror(inputFile))
        {
                perror("Error: ");
                return -1;
        }
        else
        {
                char buf[CHUNK];
                char output[CHUNK];
                size_t nread;

                outputFile = fopen("decoded.txt", "a");
                while((nread = fread(buf, 1, sizeof(buf), inputFile)) > 0)
                {
                        decode(buf,output); // decode on GPU
                        //decode_serial(buf,output); // decode on CPU
			//printf("%zu\n",nread);
                        for (int n=0;n<nread;n++)
                        	printf("%c",output[n]);
                        fwrite(output, 1, nread, outputFile);
                }
		printf("\n");
                fclose(outputFile);
                fclose(inputFile);
        }
        return 0;
}

