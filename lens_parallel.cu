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
__global__ void cudaShoot(float* device_array, int threadPerRow, float* xlens, float* ylens, float* eps,
        float rsrc2, float xsrc, float ysrc, int nlenses, int N)
{
  // What element of the array does this thread work on
    float xl, yl, xs, ys, sep2, mu;


    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    yl = YL1 + (tid / threadPerRow) * lens_scale;
    xl = XL1 + (tid % threadPerRow) * lens_scale;

    float xd, yd;
    int numuse = 0;
    float dx, dy, dr;

    xs = xl;
    ys = yl;
    for (int p = 0; p < nlenses; ++p) {
      dx = xl - xlens[p];
      dy = yl - ylens[p];
      dr = dx * dx + dy * dy;
      xs -= eps[p] * dx / dr;
      ys -= eps[p] * dy / dr;
    }


     xd = xs -xsrc;
     yd = ys -ysrc;
     sep2 = xd * xd + yd * yd;
     if (sep2 < rsrc2)
     {
         mu = sqrt (1 - sep2 / rsrc2);
         device_array[tid%threadPerRow][tid /threadPerRow] = 1.0 - ldc * (1 - mu);
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

    const float rsrc = 0.1;      // radius
    const float ldc  = 0.5;      // limb darkening coefficient
    const float xsrc = 0.0;      // x and y centre on the map
    const float ysrc = 0.0;

    const float lens_scale = 0.005;

    const int npixx = static_cast<int>(floor((XL2 - XL1) / lens_scale)) + 1;
    const int npixy = static_cast<int>(floor((YL2 - YL1) / lens_scale)) + 1;   //npixx = 801, npixy = 801
    std::cout << "# Building " << npixx << "X" << npixy << " lens image" << std::endl;


    int npitotal = npixx * npixy;
    size_t size = npitotal * sizeof(float);
    // CUDA event types used for timing execution
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
    cudaMalloc(&device_place, size);
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

    const float rsrc2 = rsrc * rsrc;

    // Launch kernel and time it
    cudaEventRecord(start, 0);


    cudaShoot<<<blocksPerGrid, threadsPerBlock>>>(device_plane, npixx, dev_xlens, dev_ylens, dev_eps, rsrc2, xsrc, ysrc, npitotal);


    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);




  float time;  // Must be a float
  cudaEventElapsedTime(&time, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  std::cout << "Kernel took: " << time << " ms" << std::endl;

  // Copy result from device memory into host memory
  cudaMemcpy(host_plane, device_plane, size, cudaMemcpyDeviceToHost);

  for (int i=0; i< npixy; i++)
      for(int j =0 ; j< npixx; j++)
      {
          lensim(i, j ) = device_plane[i][j];
      }

  // Free device memory
  cudaFree(device_plane);
  cudaFree(dev_xlens);
  cudaFree(dev_ylens);
  cudaFree(dev_eps);

  dump_array<float, 2>(lensim, "lens5.fit");

  delete[] xlens;
  delete[] ylens;
  delete[] eps;

  free(host_plane);
}
