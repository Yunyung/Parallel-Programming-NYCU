#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
extern "C"{
#include "hostFE.h"
}

__global__ void convKernel(float *inputImage, float *outputImage, float *filter,
                     const int imageHeight, const int imageWidth, const int filterWidth)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * blockDim.y + ty;
    int col = blockIdx.x * blockDim.x + tx;
    int halfFilterSize = filterWidth / 2;
    int k, l;
    float sum = 0.0f;


    __shared__ float sfilter[100]; // Assume max fitler is 10*10
    int threadIdxInBlock = (ty * blockDim.x) + tx;
    if (threadIdxInBlock < filterWidth*filterWidth) {
        sfilter[threadIdxInBlock] = filter[threadIdxInBlock];
    }
    __syncthreads();


    for (k = -halfFilterSize;k <= halfFilterSize;k++) {
        for (l = -halfFilterSize; l <= halfFilterSize; l++)
        {
            if(sfilter[(k + halfFilterSize) * filterWidth + l + halfFilterSize] != 0)
            {
                if (row + k >= 0 && row + k < imageHeight &&
                    col + l >= 0 && col + l < imageWidth)
                {
                    sum += inputImage[(row + k) * imageWidth + col + l] *
                            sfilter[(k + halfFilterSize) * filterWidth +
                                    l + halfFilterSize];
                }
            }
        }
    }

    outputImage[row * imageWidth + col] = sum;
}

extern "C"
void hostFE(int filterWidth, float *filter, int imageHeight, int imageWidth,
            float *inputImage, float *outputImage, cl_device_id *device,
            cl_context *context, cl_program *program)
{
    int filterSize = filterWidth * filterWidth * sizeof(float);
    int imageSize = imageHeight * imageWidth * sizeof(float);

    float *d_inputImage;
    float *d_outputImage;
    float *d_filter;
    cudaMalloc(&d_inputImage, imageSize);
    cudaMalloc(&d_outputImage, imageSize);
    cudaMalloc(&d_filter, filterSize);

    cudaMemcpy(d_filter, filter, filterSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_inputImage, inputImage, imageSize, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(25, 25);
    dim3 numBlocks(imageWidth / threadsPerBlock.x, imageHeight / threadsPerBlock.y);
    convKernel<<<numBlocks, threadsPerBlock>>>(d_inputImage, d_outputImage, d_filter, imageHeight, imageWidth, filterWidth);

    cudaMemcpy(outputImage, d_outputImage, imageSize, cudaMemcpyDeviceToHost);

    // cudaFree(d_filter);
    // cudaFree(d_inputImage);
    // cudaFree(d_outputImage);


}