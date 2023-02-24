#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>


#define NUM_THREADS 8
#define GROUP_SIZE 8
__global__ void mandelKernel(int *d_data,
                             int width,
                             float stepX, float stepY,
                             float lowerX, float lowerY,
                             int maxIteration,size_t pitch) {
    // To avoid error caused by the floating number, use the following pseudo code
    //

    int thisX = (blockIdx.x * blockDim.x + threadIdx.x) * GROUP_SIZE;
    int thisY = blockIdx.y * blockDim.y + threadIdx.y;
    float c_im = lowerY + thisY * stepY;
    int i;
    int* row = (int*)((char*)d_data + thisY * pitch);
    
    for (int j = 0; j < GROUP_SIZE; j++) {
      float c_re = lowerX + thisX * stepX;
      float z_re = c_re, z_im = c_im;
      
      for (i = 0; i < maxIteration; i++)
      {
        if (z_re * z_re + z_im * z_im > 4.f)
          break;

        float new_re = z_re * z_re - z_im * z_im;
        float new_im = 2.f * z_re * z_im;
        z_re = c_re + new_re;
        z_im = c_im + new_im;
      }
      
      row[thisX] = i;
      thisX++;
    }

}

// Host front-end function that allocates the memory and launches the GPU kernel
void hostFE (float upperX, float upperY, float lowerX, float lowerY, int* img, int resX, int resY, int maxIterations)
{
    float stepX = (upperX - lowerX) / resX;
    float stepY = (upperY - lowerY) / resY;
    
    int size = resX * resY * sizeof(int);
    int *h_out;
    int *d_out;
    size_t pitch;
    cudaHostAlloc((void **)&h_out, size, cudaHostAllocDefault);
    cudaMallocPitch((void **)&d_out, &pitch, sizeof(int)*resX, resY);
    dim3 block(NUM_THREADS, NUM_THREADS);
    dim3 grid(resX / (block.x*GROUP_SIZE), resY / block.y);
    
    mandelKernel<<<grid, block>>>(d_out, resX, stepX, stepY, lowerX, lowerY, maxIterations, pitch);

    cudaMemcpy2D(h_out, resX*sizeof(int), d_out, pitch, resX*sizeof(int), resY, cudaMemcpyDeviceToHost);
    memcpy(img, h_out, size);
    
    cudaFreeHost(h_out);
    cudaFree(d_out);
}
