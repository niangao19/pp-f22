#include <stdio.h>
#include <stdlib.h>
#include "hostFE.h"
#include "helper.h"

#define LOCAL_SIZE 8

void hostFE(int filterWidth, float *filter, int imageHeight, int imageWidth,
            float *inputImage, float *outputImage, cl_device_id *device,
            cl_context *context, cl_program *program)
{
    cl_int status;

    int filterSize = filterWidth * filterWidth * sizeof(float);
    int imgsize = imageHeight * imageWidth * sizeof(float);
    
    size_t global_work_size[2] = {imageWidth, imageHeight};
    size_t local_work_size[2] = {LOCAL_SIZE, LOCAL_SIZE};
    
    cl_command_queue queue = clCreateCommandQueue( *context, *device, 0, NULL );
    
    
    cl_mem inputImgMem = clCreateBuffer(*context, CL_MEM_USE_HOST_PTR, imgsize, inputImage, NULL);
    cl_mem filterMem = clCreateBuffer(*context, CL_MEM_USE_HOST_PTR, filterSize, filter, NULL);
    cl_mem outputImgMem = clCreateBuffer(*context, CL_MEM_WRITE_ONLY, imgsize, NULL, NULL);
    
    cl_kernel kernel = clCreateKernel(*program, "convolution", NULL);
    
    clSetKernelArg(kernel, 0, sizeof(cl_int), (void *) &filterWidth);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *) &filterMem);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *) &inputImgMem);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *) &outputImgMem);

    clEnqueueNDRangeKernel( queue, kernel, 2, NULL, global_work_size, local_work_size, 0 , NULL ,NULL );
    clEnqueueReadBuffer( queue,  outputImgMem, CL_TRUE, 0, imgsize, (void *)outputImage, 0, NULL, NULL);
    
//    clReleaseCommandQueue( queue );
//    clReleaseMemObject(filterMem);
//    clReleaseMemObject(inputImgMem);
//    clReleaseMemObject(outputImgMem);
//    clReleaseKernel(kernel);
}
