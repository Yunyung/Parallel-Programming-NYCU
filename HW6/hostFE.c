#include <stdio.h>
#include <stdlib.h>
#include "hostFE.h"
#include "helper.h"

void hostFE(int filterWidth, float *filter, int imageHeight, int imageWidth,
            float *inputImage, float *outputImage, cl_device_id *device,
            cl_context *context, cl_program *program)
{
    cl_int status;
    int filterSize = filterWidth * filterWidth;
    int imageSize = imageHeight * imageWidth;
    // Create a command queue
    cl_command_queue commandQueue = clCreateCommandQueue(*context, *device, 0, NULL);
    // Create memory buffers on the device
    cl_mem inputImgMem = clCreateBuffer(*context, CL_MEM_USE_HOST_PTR, imageSize*sizeof(float), inputImage, NULL);
    cl_mem filterMem = clCreateBuffer(*context, CL_MEM_USE_HOST_PTR, filterSize*sizeof(float), filter, NULL);
    cl_mem outputImgMem = clCreateBuffer(*context, CL_MEM_WRITE_ONLY, imageSize*sizeof(float), NULL, NULL);

    cl_kernel kernel = clCreateKernel(*program, "convolution", NULL);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&inputImgMem);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&outputImgMem);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&filterMem);
    clSetKernelArg(kernel, 3, sizeof(cl_int), (void *)&imageHeight);
    clSetKernelArg(kernel, 4, sizeof(cl_int), (void *)&imageWidth);
    clSetKernelArg(kernel, 5, sizeof(cl_int), (void *)&filterWidth);

    size_t global_work_size = imageSize;
    size_t local_work_size = 64; // Note : global_work_size must be divisible by local_work_size
    clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL);
    clEnqueueReadBuffer(commandQueue, outputImgMem, CL_TRUE, 0, imageSize*sizeof(float), outputImage, 0, NULL, NULL);

    // release opencl object
    // status = clReleaseCommandQueue(commandQueue);
    // status = clReleaseMemObject(inputImgMem);
    // status = clReleaseMemObject(outputImgMem);
    // status = clReleaseMemObject(filterMem);
    // status = clReleaseKernel(kernel);
}