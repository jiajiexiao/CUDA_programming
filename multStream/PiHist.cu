/*
CSC691 GPU programming
Project 4: Multi-Pi using multiple streams
Jiajie Xiao
Nov 19, 2017
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
            int i, stream_idx;

            char *buf;
            // paged-locked allocation is required when inolving streams (for pinned data that is stored in memory all the time)
            //cudaHostAlloc((void**)&buf, CHUNK*sizeof(char), cudaHostAllocDefault);
            cudaMallocHost((void**)&buf, CHUNK*sizeof(char));
            int *histc;
            cudaHostAlloc((void**)&histc, 10*sizeof(int), cudaHostAllocDefault);
            for(int i=0;i<10;i++) histc[i] = 0;


            int *histc_temp; // used for hist merging among streams and devices
            cudaHostAlloc((void**)&histc_temp, 10*sizeof(int), cudaHostAllocDefault);

            size_t nread;
           
            cudaEvent_t start_gpu, stop_gpu;
            cudaEventCreate(&start_gpu);
            cudaEventCreate(&stop_gpu);
            cudaEventRecord(start_gpu);    

            // Create two streams
            int num_streams = 2;
            cudaStream_t streams[num_streams];
            for (int stream_idx=0; stream_idx < num_streams; stream_idx++) 
                cudaStreamCreate(&(streams[stream_idx])); 

            char *dev_buf0, *dev_buf1;
            int *dev_histc0, *dev_histc1;

            cudaMalloc((void**)&dev_buf0, (int)(CHUNK/num_streams) * sizeof(char));
            cudaMalloc((void**)&dev_buf1, (int)(CHUNK/num_streams) * sizeof(char));
            cudaMalloc((void**)&dev_histc0, 10 * sizeof(int));
            cudaMalloc((void**)&dev_histc1, 10 * sizeof(int));

            cudaError_t err1, err2;
            
            int NumDigitsHistogramed = 0, Finished = 0, last; 
            while((nread = fread(buf, 1, CHUNK * sizeof(char), inputFile)) > 0 && Finished !=1)
            {
                printf("# Element loaded in CHUNK: %d\t%d\t%d\n",(int)nread, (int)(CHUNK * sizeof(char)), (int)sizeof(buf));  
                               
                if (numTruncated == -1 || NumDigitsHistogramed + (int)nread < numTruncated)
                    last = 0;
                else
                    last = 1;

                // Compute partial hist
                cudaMemcpy(dev_histc0, histc, 10 * sizeof(int), cudaMemcpyHostToDevice);
                cudaMemcpy(dev_histc1, histc, 10 * sizeof(int), cudaMemcpyHostToDevice);

                cudaMemcpyAsync(dev_buf0, buf, (int)(CHUNK/num_streams) * sizeof(char), cudaMemcpyHostToDevice, streams[0]);
                cudaMemcpyAsync(dev_buf1, buf + (int)(CHUNK/num_streams), (int)(CHUNK/num_streams) * sizeof(char), cudaMemcpyHostToDevice, streams[1]);
                //cudaMemcpyAsync(dev_buf1, buf, (int) (1.0/num_streams * CHUNK) * sizeof(char), cudaMemcpyHostToDevice, streams[1]);

                // partial hist execution 
                if(last)
                {
                    partialHist <<< 100,1000, 0, streams[0] >>> (dev_buf0,  (numTruncated - NumDigitsHistogramed)/num_streams, dev_histc0); 
                    err1 = cudaGetLastError();
                    partialHist <<< 100,1000, 0, streams[1] >>> (dev_buf1,  (numTruncated - NumDigitsHistogramed)/num_streams, dev_histc1);
                    err2 = cudaGetLastError();
                }
                else
                {
                    partialHist <<< 100,1000, 0, streams[0] >>> (dev_buf0, (int) nread/num_streams, dev_histc0); 
                    err1 = cudaGetLastError();
                    partialHist <<< 100,1000, 0, streams[1] >>> (dev_buf1, (int) nread/num_streams, dev_histc1); 
                    err2 = cudaGetLastError();
                }
                
                
                if (err1 != cudaSuccess && err2 !=cudaSuccess) 
                {
                    printf("Error: stream1 %s\n stream2 %s\n", cudaGetErrorString(err1),cudaGetErrorString(err2));
                    return -1;
                }    

                // Synchronization                    
                for (stream_idx=0;stream_idx<num_streams;stream_idx++)
                {
                    cudaStreamSynchronize(streams[stream_idx]);
                    printf("Stream Synchronized (#) %d.\n", stream_idx);
                }

                cudaDeviceSynchronize();

                // Merge partial hist
                //cudaMemcpy(histc_temp, dev_histc0, 10 * sizeof(int), cudaMemcpyDeviceToHost);
                cudaMemcpyAsync(histc_temp, dev_histc0, 10 * sizeof(int), cudaMemcpyDeviceToHost,streams[0]);
                for (i=0;i<10;i++) 
                {
                    histc[i]+=histc_temp[i];
                    //printf("%d\t%d\n",i,histc_temp[i]);
                }
                cudaStreamSynchronize(streams[0]);
                cudaMemcpyAsync(histc_temp, dev_histc1, 10 * sizeof(int), cudaMemcpyDeviceToHost,streams[1]);
                for (i=0;i<10;i++) 
                {
                    histc[i]+=histc_temp[i];
                    //printf("%d\t%d\n",i,histc_temp[i]);            
                }
                cudaStreamSynchronize(streams[1]);

                // // cupy memory from gpu 1 to gpu0 using synchronized one instead of the asynchronized one
                // cudaMemcpyPeer(histc_temp, 0, dev_histc2, 1, 10 * sizeof(int));
                // for (i=0;i<10;i++) 
                //     histc[i]+=histc_temp[i];
                // cudaMemcpyPeer(histc_temp, 0, dev_histc3, 1, 10 * sizeof(int));
                // for (i=0;i<10;i++) 
                //     histc[i]+=histc_temp[i];
                
                if(last)
                {
                    NumDigitsHistogramed = numTruncated;
                    Finished = 1;
                }    
                else
                {
                    NumDigitsHistogramed += (int)nread;
                }
            }
            cudaEventRecord(stop_gpu);

            cudaStreamDestroy(streams[0]);
            cudaStreamDestroy(streams[1]);
            cudaFree(dev_histc0);
            cudaFree(dev_histc1);
            cudaFree(dev_buf0);
            cudaFree(dev_buf1);

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

