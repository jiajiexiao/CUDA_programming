/*
CSC691 GPU programming
Project 3: Pi Time
Jiajie Xiao
Oct 23, 2017
*/

#include<stdio.h>
#include <curand.h> // cuRAND host and device header files
#include <curand_kernel.h> 

//#define numsamples 1000000
//#define numThreadPerBlock 1024


// kernel to initialize the random states
__global__ void random_init (unsigned int seed, curandState_t *states)
{
	curand_init(seed, threadIdx.x, 0, &states[blockIdx.x * blockDim.x + threadIdx.x]);
}


// Generate a random point  
__global__ void random_sample(curandState_t * states, unsigned int * SuccessLanded, unsigned int numLoops)
{
	float x, y;
	unsigned int n, idx;
	idx = blockIdx.x * blockDim.x + threadIdx.x;
	SuccessLanded[idx] = 0;

	for (n=0; n<numLoops; n++)
	{
		x = curand_uniform(&states[idx]);
		y = curand_uniform(&states[idx]);
		//printf("%f\t%f\n", x,y);
		if (x*x+y*y<1) 
			SuccessLanded[idx]++;	
	}
}


int main(int argc, char **argv)
{
	int numsamples;
	if (argc < 2)
    {
    	printf("Please type in number of iterations to compute Pi.\n");
        return -1;
    }
    else
    {
    	numsamples = atoi(argv[1]);
    	if(numsamples<1)
    	{
    		printf("Please type in positive itergers.\n");
    		return -1;
    	}
    }
    int numThreadPerBlock = numsamples>1000?1000:numsamples;


	unsigned int numBlocks, numLoops;
	numBlocks = numsamples/numThreadPerBlock;
	if(numBlocks>100)
		numBlocks =100;
	numLoops = numsamples/(numBlocks*numThreadPerBlock);

	unsigned int *SuccessLanded, *dev_SuccessLanded;
	SuccessLanded = (unsigned int *) malloc(numThreadPerBlock*numBlocks * sizeof(unsigned int));
	cudaMalloc((void**)&dev_SuccessLanded, numThreadPerBlock*numBlocks * sizeof(unsigned int));
	cudaMemcpy(dev_SuccessLanded, SuccessLanded, numThreadPerBlock*numBlocks * sizeof(unsigned int), cudaMemcpyHostToDevice);

	curandState_t *states;
	cudaMalloc((void**) &states, numThreadPerBlock*numBlocks * sizeof(curandState_t));


	// initialize all of the random states on the GPU
	random_init<<<numBlocks,numThreadPerBlock>>>(time(NULL),states);

	// Perform Monte Carlo sampling
	random_sample<<<numBlocks,numThreadPerBlock>>> (states, dev_SuccessLanded, numLoops);

	cudaEvent_t start_gpu, stop_gpu;
	cudaEventCreate(&start_gpu);
	cudaEventCreate(&stop_gpu);

	cudaEventRecord(start_gpu);
	// initialize all of the random states on the GPU
	random_init<<<numBlocks,numThreadPerBlock>>>(time(NULL),states);

	// Perform Monte Carlo sampling
	random_sample<<<numBlocks,numThreadPerBlock>>> (states, dev_SuccessLanded, numLoops);
	//cudaThreadSynchronize();
	cudaEventRecord(stop_gpu);
	
	cudaMemcpy(SuccessLanded, dev_SuccessLanded, numThreadPerBlock*numBlocks * sizeof(unsigned int), cudaMemcpyDeviceToHost);
	cudaFree(dev_SuccessLanded);

	cudaEventSynchronize(stop_gpu);
	float milliseconds_gpu = 0;
    cudaEventElapsedTime(&milliseconds_gpu, start_gpu, stop_gpu);

	clock_t start_cpu, stop_cpu;
    start_cpu = clock();
	unsigned int numPointsLanded = 0;
	for (unsigned int n=0; n<(numThreadPerBlock*numBlocks); n++)
	{
		numPointsLanded += SuccessLanded[n];	
	}
	stop_cpu = clock();
    float milliseconds_cpu = 1000.0*(stop_cpu-start_cpu)/(float)CLOCKS_PER_SEC;


	unsigned int actualSample = numLoops* numBlocks * numThreadPerBlock;
	printf("%d\t%d\t%d\t%d.\n", numPointsLanded, actualSample, numBlocks, numLoops);
	float computedPi = 4.0*numPointsLanded/(actualSample);
	printf("The computed Pi has a value of %.6f.\n", computedPi);

	FILE *file;
    file = fopen("results.txt", "a");
    if (file == NULL)
        printf("Error open file to save the result file");
    else
    {
        fprintf(file, "%d\t%e\t%e\t%e\n", actualSample, computedPi, milliseconds_gpu, milliseconds_cpu);
        fclose(file);
    }
	
	return 0;
}
