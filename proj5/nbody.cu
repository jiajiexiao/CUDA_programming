#include <cstdio>
#include <cstdlib>
#include <cmath>

#define N 9999  // number of bodies
#define MASS 0  // row in array for mass
#define X_POS 1 // row in array for x position
#define Y_POS 2 // row in array for y position
#define Z_POS 3 // row in array for z position
#define X_VEL 4 // row in array for x velocity
#define Y_VEL 5 // row in array for y velocity
#define Z_VEL 6 // row in array for z velocity
#define G 10    // "gravitational constant" (not really)

float dt = 0.05; // time interval

// JJ: one could compute the distance matrix using tiles first in pararrel, then compute forces, and finnaly update the velocities and positions. In this naive project, I chose not to implement it.

// each thread computes new position of one body 
__global__ void nbody(float *dev_body, float dt) {

  // Index for body i
  int i = blockDim.x*blockIdx.x + threadIdx.x;
  if(i<N)  // if index is larger than number of bodies, no need to calculate.
  {
  	float x_diff, y_diff, z_diff, r;
	float Fx_dir=0, Fy_dir=0, Fz_dir=0;

	for (int j=0; j<N && j!=i; j++) // force on body i due to body j
	{
		x_diff = dev_body[j*7+X_POS] - dev_body [i*7+X_POS];  // difference in x direction
		y_diff = dev_body[j*7+Y_POS] - dev_body [i*7+Y_POS];  // difference in y direction
		z_diff = dev_body[j*7+Z_POS] - dev_body [i*7+Z_POS];  // difference in z direction


		// calculate distance between i and j (r)
		r = sqrt(x_diff * x_diff + y_diff * y_diff + z_diff * z_diff);
		// force between bodies i and x
		float F = G * dev_body[i*7+MASS] * dev_body[j*7+MASS] / r; //JJ: It should be r^2 on my opinion. The same for the following.

		if (r > 10.0)
		{
		Fx_dir += F * x_diff / r;  // resolve forces in x and y directions
		Fy_dir += F * y_diff / r;  // and accumulate forces
		Fz_dir += F * z_diff / r;  // 
		}
		else
		{
		// if too close, anti-gravitational force
		Fx_dir -= F * x_diff / r;  // resolve forces in x and y directions
		Fy_dir -= F * y_diff / r;  // and accumulate forces
		Fz_dir -= F * z_diff / r;  // 
		}
	}

	// update velocities
	dev_body[i*7+X_VEL] += Fx_dir * dt / dev_body[i*7+MASS];
	dev_body[i*7+Y_VEL] += Fy_dir * dt / dev_body[i*7+MASS];
	dev_body[i*7+Z_VEL] += Fz_dir * dt / dev_body[i*7+MASS];
	// update positions
	dev_body[i*7+X_POS] += dev_body[i*7+X_VEL] * dt;
	dev_body[i*7+Y_POS] += dev_body[i*7+Y_VEL] * dt;
	dev_body[i*7+Z_POS] += dev_body[i*7+Z_VEL] * dt;

  } 
}


int main(int argc, char **argv) {

  float *body; // host data array of bodies
  float *dev_body; // device data array of bodies

  int tmax = 0;

  if (argc != 2) {
    fprintf(stderr, "Format: %s { number of timesteps }\n", argv[0]);
    exit (-1);
  }

  tmax = atoi(argv[1]);
  printf("Requested Timesteps: %d.\n", tmax);

  // allocate memory size for the body
  int bodysize = N * 7 * sizeof(float);    
  body = (float *)malloc(bodysize);
  cudaMalloc((void**) &dev_body, bodysize);

  // assign each body a random position
  for (int i = 0; i < N; i++) {
    body[i * 7 + MASS] = 10;
    body[i * 7 + X_POS] = drand48() * 100.0;
    body[i * 7 + Y_POS] = drand48() * 100.0;
    body[i * 7 + Z_POS] = drand48() * 100.0;
    body[i * 7 + X_VEL] = 0.0;
    body[i * 7 + Y_VEL] = 0.0;
    body[i * 7 + Z_VEL] = 0.0;
  }

  // print out initial positions in PDB format
  FILE * fp;
  fp = fopen ("NBody.pdb", "w");

  fprintf(fp, "MODEL %8d\n", 0);
  for (int i = 0; i < N; i++) {
    fprintf(fp, "%s%7d  %s %s %s%4d    %8.3f%8.3f%8.3f  %4.2f  %4.3f\n",
           "ATOM", i+1, "CA ", "GLY", "A", i+1, body[i * 7 + X_POS], body[i * 7 + Y_POS], body[i * 7 + Z_POS], 1.00, 0.00);
    fflush(fp);
  }
  fprintf(fp, "TER\nENDMDL\n");

  // step through each time step
  for (int t = 0; t < tmax; t++) {

    // copy nbody info over to GPU
    cudaMemcpy(dev_body, body, bodysize, cudaMemcpyHostToDevice);

    dim3 blockDim(1024);
    dim3 gridDim((int)ceil(N / blockDim.x) + 1);

    // run nbody calculation
    nbody<<<gridDim, blockDim>>>(dev_body, dt);
    cudaThreadSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
    {
		printf("Error: %s when t=%d\n", cudaGetErrorString(err),t);
		return -1;
    }   

    // copy nbody info back to CPU
    cudaMemcpy(body, dev_body, bodysize, cudaMemcpyDeviceToHost);

    // print out positions in PDB format
    fprintf(fp, "MODEL %8d\n", t+1);
    for (int i = 0; i < N; i++) {
      int error = fprintf(fp, "%s%7d  %s %s %s%4d    %8.3f%8.3f%8.3f  %4.2f  %4.3f\n","ATOM", i+1, "CA ", "GLY", "A", i+1, body[i * 7 + X_POS], body[i * 7 + Y_POS], body[i * 7 + Z_POS], 1.00, 0.00);
      if (error ==-1) 
      {
      	printf("printf error when t=%d,i=%d\n", t+1,i);
      	break;
      }
      else
      	fflush(fp);
    }
    fprintf(fp, "TER\nENDMDL\n");

  }  // end of time period loop

  fclose(fp);
  free(body);
  cudaFree(dev_body);
}

