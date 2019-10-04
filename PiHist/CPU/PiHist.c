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

int partialHist(char *input, int len ,int *hist)
{
	for (int idx=0;idx<len;idx++)	
	{
		//printf("%c\t%d\n",input[idx],input[idx]-'0');
		hist[input[idx]-'0']++; // Treat the input numbers as indices
	}
	return 0;
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

                clock_t start_cpu, stop_cpu;
    			float milliseconds_cpu = 0;

    			int NumDigitsHistogramed = 0, Finished = 0; // iteration for streaming data
                while((nread = fread(buf, 1, sizeof(buf), inputFile)) > 0 && Finished != 1)
                {
 					start_cpu = clock();
 					// if no specified number of digits or the streaming is not larger than the specified number yet
 					if (numTruncated == -1 || NumDigitsHistogramed + (int)nread < numTruncated)
 					{
 						partialHist(buf,(int)nread, histc);
 						NumDigitsHistogramed += (int)nread;
 					}
 					else // The streaming is approacing the number of selected
 					{
 						partialHist(buf,numTruncated - NumDigitsHistogramed, histc);
 						NumDigitsHistogramed = numTruncated;
 						Finished = 1;
 					}
 					stop_cpu = clock();
    				milliseconds_cpu += 1000.0*(stop_cpu-start_cpu)/(float)CLOCKS_PER_SEC;
                }
                fclose(inputFile);
				printf("The histograming calculation time (ms): %f\n", milliseconds_cpu);

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
