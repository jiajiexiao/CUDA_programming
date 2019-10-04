/*
CSC691 GPU programming
Project 3: Pi Time
Jiajie Xiao
Oct 23, 2017
*/
#include<stdio.h>
#include <stdlib.h> // srand, rand
#include <time.h> 
//#define numsamples 100000000

// Generate a random point  
int generatePoint(float r, float *x, float *y)
{
	// srand (time(NULL));
	*x = r*rand()/RAND_MAX;
	*y = r*rand()/RAND_MAX;
	return 0;
}

// Determine if the point is under the curve (quarter-circle)
int determinePoint(float r, float x, float y)
{
	if (x*x+y*y<r*r)
		return 1;
	else
		return 0;
}

int main(int argc, char **argv)
{
	unsigned int numsamples;
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

	float r=1.0;
	float x, y;
	unsigned int numPointsLanded = 0;

	clock_t start_cpu, stop_cpu;
    start_cpu = clock();

	srand (time(NULL)); // initialize random seed

	for (unsigned int n=0; n<numsamples; n++)
	{
		generatePoint(r, &x, &y);
		// printf("%f, %f\n",x, y);
		numPointsLanded += determinePoint(r, x, y);

	}
	stop_cpu = clock();
    float milliseconds_cpu = 1000.0*(stop_cpu-start_cpu)/(float)CLOCKS_PER_SEC;

	//printf("%d.\n", numPointsLanded);
	float computedPi = 4.0*numPointsLanded/numsamples/(r*r);
	printf("The computed Pi has a value of %.6f.\n", computedPi);

	FILE *file;
    file = fopen("results.txt", "a");
    if (file == NULL)
        printf("Error open file to save the result file");
    else
    {
        fprintf(file, "%d\t%e\t%e\n", numsamples, computedPi, milliseconds_cpu);
        fclose(file);
    }
	
	
	return 0;
}
