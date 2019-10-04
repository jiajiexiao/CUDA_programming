/*
CSC691 GPU programming
Project 2: In the Interest of Time
Naive GPU version
Jiajie Xiao
Oct 8, 2017
*/

#include <stdio.h> 
#include <stdlib.h>
#include <time.h>

void gen_flattened_array(float *array, int rowSize, int colSize)
{
    int i, j;
	time_t t;
	/* Intializes random number generator */
   	srand((unsigned) time(&t));
   	for (i=0;i<rowSize;i++)
   	{
   		for (j=0;j<colSize;j++)
   		{
   			array[i*colSize+j]=1.0*rand()/RAND_MAX;
   		}
   	}
}

void output_array(float *array, int rowSize, int colSize)
{
    int i, j;
    printf("%s\t%s\t%s\n", "RowID", "ColID", "Value");
    for (i=0; i<rowSize; i++)
    {
    	for (j=0;j<colSize;j++)
    	{
    		 printf("%d\t%d\t%f\n", i, j, array[i*colSize+j]);
    	}
    }
}

void save_array(float *array, int rowSize, int colSize, FILE *fp)
{
    int i, j;
    for (i=0; i<rowSize; i++)
    {
        for (j=0;j<colSize;j++)
        {
            fprintf(fp, "%d\t%d\t%f\n", i, j, array[i*colSize+j]);
        }
    }
}

__global__ void compute_matixMult(float *a, float *b, float *c, int rowSize_a, int colSize_a, int colSize_b) // colSize_a should be equal to rowSize_b
{
    int k;
    float sum = 0.0;
    int col = threadIdx.x + blockDim.x * blockIdx.x;
    int row = threadIdx.y + blockDim.y * blockIdx.y;

	if (col < rowSize_a && row < colSize_b)
    {
        for (k = 0; k < colSize_a; k++)
        {
            sum += a[row*colSize_a+k]*b[k*colSize_b+col];
        }
        c[row*colSize_b+col] = sum;
    }
}

int main(int argc, char **argv)
{
    int rowSize_A, colSize_A, rowSize_B, colSize_B, rowSize_C, colSize_C;

    if (argc < 2)
    {
        printf("At least one varaible for matrix dimensions is required.");
        return -1;
    }
    else if (argc == 2)
    {	
    	rowSize_A = atoi(argv[1]);
    	colSize_A = atoi(argv[1]);
    	rowSize_B = atoi(argv[1]);
    	colSize_B = atoi(argv[1]);
    	printf("Two random %dX%d matrices are going to be generated for the multiplication.\n", rowSize_A, colSize_A);
    }
    else
    {    	
    	rowSize_A = atoi(argv[1]);
    	colSize_A = atoi(argv[2]);
    	rowSize_B = atoi(argv[3]);
    	colSize_B = atoi(argv[4]);

    	if (colSize_A == rowSize_B)
    		printf("A random %dX%d matrix A and %dX%d matrix B are going to be generated for the multiplication.\n", rowSize_A, colSize_A, rowSize_B, colSize_B);
    	else
    	{
    		printf("Error input: the number of columns of the first matrix should be equal to the number of rows of te second matrix");
    		return -1;
    	}
    }

    /* Generated two random matrices for multiplication */
    rowSize_C = rowSize_A; 
    colSize_C = colSize_B;
    float matrix_A[rowSize_A*colSize_A], matrix_B[rowSize_B*colSize_B], matrix_C[rowSize_C*colSize_C]; 
    gen_flattened_array(matrix_A, rowSize_A, colSize_A);
    gen_flattened_array(matrix_B, rowSize_B, colSize_B);

    /* Malloc memory for device variables */
    float *dev_matrix_A, *dev_matrix_B, *dev_matrix_C; 
    cudaMalloc((void **) & dev_matrix_A, rowSize_A*colSize_A*sizeof(float));
    cudaMalloc((void **) & dev_matrix_B, rowSize_B*colSize_B*sizeof(float));
    cudaMalloc((void **) & dev_matrix_C, rowSize_C*colSize_C*sizeof(float));

    /* Computate the multiplication */
    dim3 dimGrid(1,1);
    dim3 dimBlock(colSize_A, colSize_A);

    compute_matixMult<<<dimGrid, dimBlock>>>(matrix_A, matrix_B, matrix_C, rowSize_A, colSize_B, rowSize_B);
    cudaThreadSynchronize();

    clock_t start, stop;
    start = clock();
    compute_matixMult<<<dimGrid, dimBlock>>>(matrix_A, matrix_B, matrix_C, rowSize_A, colSize_B, rowSize_B);
    cudaThreadSynchronize();
    stop = clock();

    /* transfer data from Device to Host and free device memory */
    cudaMemcpy(matrix_C, dev_matrix_C, rowSize_C*colSize_C*sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(dev_matrix_A);
    cudaFree(dev_matrix_B);
    cudaFree(dev_matrix_C);

    float time = (stop-start)/(float)CLOCKS_PER_SEC;
    printf("The product is computed.\n Execution time is %f seconds\n", time);

    /* Output */
    //printf("Matrix A\n");
    //output_array(matrix_A, rowSize_A, colSize_A);
    //printf("Matrix B\n");
    //output_array(matrix_B, rowSize_B, colSize_B);
    //printf("Product Matrix C\n");    
    //output_array(matrix_C, rowSize_A, colSize_B);

    FILE *file_product, *file_time;
    file_product = fopen("product.txt", "w");
    if (file_product == NULL)
        printf("Error open file to save the product matrix");
    else
    {
        save_array(matrix_C, rowSize_A, colSize_B, file_product);
        fclose(file_product);
    }


    file_time = fopen("time.txt", "a");
    if (file_time == NULL)
        printf("Error open file to save the product matrix");
    else
    {
        fprintf(file_time, "%d\t%f\n", colSize_A, time);
        fclose(file_time);
    }

    return 1;

}
