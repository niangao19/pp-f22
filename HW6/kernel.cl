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

