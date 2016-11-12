/* CUDA timing example

   To compile: nvcc -o testprog2 testprog2.cu

 */
#include <iostream>
#include <string>
#include <cmath>

#include <cuda.h>
#include "lenses.h"
#include "arrayff.hxx"


// Global variables! Not nice style, but we'll get away with it here.

// Boundaries in physical units on the lens plane
const float WL  = 2.0;
const float XL1 = -WL;
const float XL2 =  WL;
const float YL1 = -WL;
const float YL2 =  WL;


// Kernel that executes on the CUDA device. This is executed by ONE
// stream processor
__global__ void cudaShoot(float* device_array, float* xlens, float* ylens, float* eps, int nlenses, int lens_scale, int ntotal)
{
  // What element of the array does this thread work on
    const float rsrc = 0.1;      // radius
    const float ldc  = 0.5;      // limb darkening coefficient
    const float xsrc = 0.0;      // x and y centre on the map
    const float ysrc = 0.0;
//    float xl, yl, xs, ys, sep2, mu;

//    float xd, yd;
//    float dx, dy, dr;

    const float rsrc2 = rsrc * rsrc;

    extern __shared__ float yl[];
    extern __shared__ float xl[];
    extern __shared__ float xs[];
    extern __shared__ float ys[];
    extern __shared__ float dx[];
    extern __shared__ float dy[];
    extern __shared__ float dr[];
    extern __shared__ float xd[];
    extern __shared__ float yd[];
    extern __shared__ float sep2[];
    extern __shared__ float mu[];

    int tidx = blockDim.x * blockIdx.x + threadIdx.x;

    if (tidx < ntotal)

    {

        yl[tidx] = YL1 + (tidx / blockDim.x) * lens_scale;
        xl[tidx] = XL1 + (tidx % blockDim.x) * lens_scale;

        xs[tidx] = xl[tidx];
        ys[tidx] = yl[tidx];

        for (int p = 0; p < nlenses; ++p) {
          dx[tidx] = xl[tidx] - xlens[p];
          dy[tidx] = yl[tidx] - ylens[p];
          dr[tidx] = dx[tidx] * dx[tidx] + dy[tidx] * dy[tidx];
          xs[tidx] -= eps[p] * dx[tidx] / dr[tidx];
          ys[tidx] -= eps[p] * dy[tidx] / dr[tidx];
        }


        xd[tidx] = xs[tidx] -xsrc;
        yd[tidx] = ys[tidx] -ysrc;
        sep2[tidx] = xd[tidx] * xd[tidx] + yd[tidx] * yd[tidx];

        if (sep2[tidx] < rsrc2)
        {
            mu[tidx] = sqrt (1 - sep2[tidx] / rsrc2);
            //device_array[tx + ty * size] = 1.0 - ldc * (1 - mu);
            //__syncthreads();
            device_array[tidx] = 1.0 - ldc * (1 - mu[tidx]);
        }
    }





}

// main routine that executes on the host
int main(void)
{
    float* xlens;
    float* ylens;
    float* eps;
    const int nlenses = set_example_1(&xlens, &ylens, &eps);
    std::cout << "# Simulating " << nlenses << " lens system" << std::endl; //nelnses = 1


    const float lens_scale = 0.005;

    const int npixx = static_cast<int>(floor((XL2 - XL1) / lens_scale)) + 1;
    const int npixy = static_cast<int>(floor((YL2 - YL1) / lens_scale)) + 1;   //npixx = 801, npixy = 801
    std::cout << "# Building " << npixx << "X" << npixy << " lens image" << std::endl;


    long npitotal = npixx * npixy;
    size_t size = npitotal * sizeof(float);


    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Allocate in HOST memory
    float* host_plane = (float*)malloc(size);


    // Initialize vectors
    for (int i = 0; i < npitotal; ++i) {
        host_plane[i] = 0;
    }


    // Allocate in DEVICE memory
    float *device_plane, *dev_xlens, *dev_ylens, *dev_eps;
    cudaMalloc(&device_plane, size);
    cudaMalloc(&dev_xlens, nlenses);
    cudaMalloc(&dev_ylens, nlenses);
    cudaMalloc(&dev_eps, nlenses);



    // Copy vectors from host to device memory
    //cudaMemcpy(device_plane, host_plane, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_xlens, xlens, nlenses, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_ylens, ylens, nlenses, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_eps, eps, nlenses, cudaMemcpyHostToDevice);



    // Set up layout of kernel grid
    int threadsPerBlock = 1024;
    int blocksPerGrid = (npitotal + threadsPerBlock - 1) / threadsPerBlock;


    // Put the lens image in this array
    Array<float, 2> lensim(npixy, npixx);


    // Launch kernel and time it
    cudaEventRecord(start, 0);

    std::cout<<"blocksPerGrid="<<blocksPerGrid<<"\nthreadsPerBlock="<<threadsPerBlock<<"\n nlenses="<<nlenses<<"\n lens_scale="<<lens_scale<<" \n npitotal="<<npitotal<<std::endl;
    cudaShoot<<<blocksPerGrid, threadsPerBlock, size>>>(device_plane, dev_xlens, dev_ylens, dev_eps, nlenses, lens_scale, npitotal);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);



  float time;  // Must be a float
  cudaEventElapsedTime(&time, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  std::cout << "Kernel took: " << time << " ms" << std::endl;

  // Copy result from device memory into host memory
  cudaMemcpy(host_plane, device_plane, size, cudaMemcpyDeviceToHost);




  // Free device memory
  cudaFree(device_plane);
  cudaFree(dev_xlens);
  cudaFree(dev_ylens);
  cudaFree(dev_eps);



  for (int i=0; i< npixy; i++)
  {
      for(int j =0 ; j< npixx; j++)
      {
          lensim(i, j) = host_plane[i*npixx+j];
      }
  }
  dump_array<float, 2>(lensim, "lens1.fit");

  delete[] xlens;
  delete[] ylens;
  delete[] eps;


  free(host_plane);
}
