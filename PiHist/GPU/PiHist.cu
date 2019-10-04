/*
CSC691 GPU programming
Project 3: Pi Time
Jiajie Xiao
Oct 23, 2017
*/


#include <stdio.h> 
#include <stdlib.h>
#include <time.h>
#define CHUNK 100000 

__global__ void partialHist(char *input, int len, int *hist)
{
    int i = threadIdx.x + blockDim.x*blockIdx.x;
    int number = input[i]-'0';
    //printf("%c\t%d\t%d\n", input[i], number, len);

    if (i<len)
    {
        // printf("%d\t",i);
        __shared__ int partial_sum[10];

         // thread 0 is responsible for initializing partial_sum
        if (threadIdx.x == 0)
        {
            for (i=0;i<10;i++)
                partial_sum[i] = 0;
        } 
        __syncthreads(); // each thread updates the partial sum
        atomicAdd(&partial_sum[number], 1);
        //printf("%d\t%d\n",number, partial_sum[number]);
        __syncthreads();

        // thread 0 updates the total sum
        if (threadIdx.x == 0)
        {
            for (i=0;i<10;i++)
                atomicAdd(&hist[i], partial_sum[i]); 
        }          
    }	
}

int main(int argc, char **argv)
{
        FILE *inputFile, *outputFile; 
        int numTruncated = -1;
        if (argc < 2)
        {
            printf("An input file name is required.");
            return -1;
        }
        else if (argc >2)
        {
            numTruncated = atoi(argv[2]);
            if (numTruncated<1)
            {
                printf("Please type positive number of digits to be evaluated.\n");
                return -1;
            }
        }
                
        inputFile = fopen(argv[1],"r");
        if (ferror(inputFile))
        {
                perror("Error: ");
                return -1;
        }
        else
        {
                char buf[CHUNK];
                int histc[10] = {0,0,0,0,0,0,0,0,0,0};
                size_t nread;

                int *dev_histc;
                cudaMalloc((void**)&dev_histc, 10 * sizeof(int));
                cudaMemcpy(dev_histc, histc, 10 * sizeof(int), cudaMemcpyHostToDevice);

                char *dev_buf;
                cudaMalloc((void**)&dev_buf, CHUNK * sizeof(char));                
                cudaEvent_t start_gpu, stop_gpu;
                cudaEventCreate(&start_gpu);
                cudaEventCreate(&stop_gpu);

                cudaEventRecord(start_gpu);
                int NumDigitsHistogramed = 0, Finished = 0; 
                while((nread = fread(buf, 1, sizeof(buf), inputFile)) > 0 && Finished !=1)
                { 

                    cudaMemcpy(dev_buf, buf, CHUNK * sizeof(char), cudaMemcpyHostToDevice);
                	//printf("%d\n",(int)nread);                    
                    if (numTruncated == -1 || NumDigitsHistogramed + (int)nread < numTruncated)
                    {
                        partialHist <<< 100,1000 >>> (dev_buf, (int) nread, dev_histc); 
                        NumDigitsHistogramed += (int)nread;
                    }
                    else // The streaming is approacing the number of selected
                    {
                        partialHist <<< 100,1000 >>> (dev_buf, numTruncated - NumDigitsHistogramed, dev_histc); 
                        NumDigitsHistogramed = numTruncated;
                        Finished = 1;
                    }
                }
                cudaEventRecord(stop_gpu);

                cudaMemcpy(histc, dev_histc, 10 * sizeof(int), cudaMemcpyDeviceToHost);
                cudaFree(dev_histc);
                cudaFree(dev_buf);

                cudaEventSynchronize(stop_gpu);
                float milliseconds_gpu = 0;
                cudaEventElapsedTime(&milliseconds_gpu, start_gpu, stop_gpu);

                fclose(inputFile);
				printf("The histograming calculation time (ms): %f\n", milliseconds_gpu);

				outputFile = fopen("hist.txt", "a");
				if (ferror(inputFile))
       			{
                	perror("Error: ");
                	return -1;
        		}
       			else
        		{
					fprintf(outputFile, "%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n", NumDigitsHistogramed, histc[0], histc[1], histc[2], histc[3], histc[4], histc[5], histc[6], histc[7], histc[8], histc[9] );
                    fclose(outputFile);
        		}
        }
        return 0;
}
