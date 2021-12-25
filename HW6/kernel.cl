__kernel void convolution(const __global float *inputImage, __global float *outputImage, __constant float *filter,
                          const int imageHeight, const int imageWidth, const int filterWidth) 
{
   int idx = get_global_id(0);
   int row = idx / imageWidth;
   int col = idx % imageWidth;
   int halfFilterSize = filterWidth / 2;
   int k, l;
   float sum = 0.0f;

   for (k = -halfFilterSize;k <= halfFilterSize;k++) {
       for (l = -halfFilterSize; l <= halfFilterSize; l++)
        {
            if(filter[(k + halfFilterSize) * filterWidth + l + halfFilterSize] != 0)
            {
                if (row + k >= 0 && row + k < imageHeight &&
                    col + l >= 0 && col + l < imageWidth)
                {
                    sum += inputImage[(row + k) * imageWidth + col + l] *
                            filter[(k + halfFilterSize) * filterWidth +
                                    l + halfFilterSize];
                }
            }
        }
   }
   
   outputImage[row * imageWidth + col] = sum;
}
