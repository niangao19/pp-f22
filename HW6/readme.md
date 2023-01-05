# 平行程式作業-hw6
###### tags:  `HW` `PP`
## Q1
>Explain your implementation. How do you optimize the performance of convolution?

在HostFE.c中，我使用了CL_MEM_USE_HOST_PTR來傳遞data到device中。
並且將local size設為8
```cpp=
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
    
    clReleaseCommandQueue( queue );
    clReleaseMemObject(filterMem);
    clReleaseMemObject(inputImgMem);
    clReleaseMemObject(outputImgMem);
    clReleaseKernel(kernel);
}
```

在kernel function參考助教給的code並改為以下
```cpp=
__kernel void convolution( int filterWidth, __constant float *filter, __global float *inputImage, __global float *outputImage)
{
    // Iterate over the rows of the source image
    int halffilterSize = filterWidth / 2;
    int imageHeight = get_global_size(1);
    int imageWidth  = get_global_size(0);
    
    int i = get_global_id(1);
    int j = get_global_id(0);

    float sum;
    int k, l;
    sum = 0; // Reset sum for new source pixel
    // Apply the filter to the neighborhood
    for (k = -halffilterSize; k <= halffilterSize; k++)
    {
        for (l = -halffilterSize; l <= halffilterSize; l++)
        {
            if (i + k >= 0 && i + k < imageHeight &&
                j + l >= 0 && j + l < imageWidth)
            {
                sum += inputImage[(i + k) * imageWidth + j + l] *
                       filter[(k + halffilterSize) * filterWidth +
                              l + halffilterSize];
            }
        }
    }
    
    outputImage[i * imageWidth + j] = sum;
}
```





[reference1](https://www.cnblogs.com/mikewolf2002/archive/2012/09/05/2671261.html)
[reference2](https://www.cnblogs.com/mikewolf2002/archive/2012/09/07/2675634.html)