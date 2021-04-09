
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include "string.h"
#include <unistd.h>

// #include <io.h>
#define DEFAULT_THRESHOLD 4000
#define DEFAULT_FILENAME "mountains.ppm"

void write_ppm(char* filename, int xsize, int ysize, int maxval, int* pic);
unsigned int* read_ppm(char* filename, int& xsize, int& ysize, int& maxval);

/*
 (j - 1, i - 1) (j, i - 1) (j + 1, i - 1)
 (j - 1, i)     (j, i)     (j + 1, i)
 (j - 1, i + 1) (j, i + 1) (j + 1, i + 1)
*/

__global__ void sobel_kernel_naive(const unsigned int* inputVec, unsigned int* outVec, int width, int height)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < 1 || j < 1 || i >= (height - 1) || j >= (width - 1))
    {
        return;
    }

    int offset = i * width + j;
    unsigned int sum1 = inputVec[width * (i - 1) + j + 1] - inputVec[width * (i - 1) + j - 1] + 2 * inputVec[width * (i)+j + 1] - 2 * inputVec[width * (i)+j - 1] + inputVec[width * (i + 1) + j + 1] - inputVec[width * (i + 1) + j - 1];

    unsigned int sum2 = inputVec[width * (i - 1) + j - 1] + 2 * inputVec[width * (i - 1) + j] + inputVec[width * (i - 1) + j + 1] - inputVec[width * (i + 1) + j - 1] - 2 * inputVec[width * (i + 1) + j] - inputVec[width * (i + 1) + j + 1];
    unsigned int magnitude = sum1 * sum1 + sum2 * sum2;

    if (magnitude > DEFAULT_THRESHOLD)
    {
        outVec[offset] = 255;
    }
}

__global__ void sobel_kernel_register_reuse(const unsigned int* inputVec, unsigned int *outVec, int width, int height)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < 1 || j < 1 || i >= (height - 1) || j >= (width - 1))
    {
        return;
    }

    int offset = i * width + j;
    unsigned int topLeft = inputVec[width * (i - 1) + j - 1];
    unsigned int topRight = inputVec[width * (i - 1) + j + 1];
    unsigned int bottomLeft = inputVec[width * (i + 1) + j - 1];
    unsigned int bottomRight = inputVec[width * (i + 1) + j + 1];

    unsigned int sum1 = topRight
        - topLeft
        + 2 * inputVec[width * (i)+j + 1] 
        - 2 * inputVec[width * (i)+j - 1] 
        + bottomRight
        - bottomLeft;
    unsigned int sum2 = topLeft
        + 2 * inputVec[width * (i - 1) + j] 
        + topRight
        - bottomLeft
        - 2 * inputVec[width * (i + 1) + j] 
        - bottomRight;
    unsigned int magnitude = sum1 * sum1 + sum2 * sum2;

    if (magnitude > DEFAULT_THRESHOLD)
    {
        outVec[offset] = 255;
    }
}

/*
 (j - 1, i - 1) (j, i - 1) (j + 1, i - 1)
 (j - 1, i)     (j, i)     (j + 1, i)
 (j - 1, i + 1) (j, i + 1) (j + 1, i + 1)
*/
__global__ void sobel_kernel_shared_mem_reuse(const unsigned int* inputVec, unsigned int *outVec, int width, int height)
{
    int j = blockIdx.x * 14 + threadIdx.x;
    int i = blockIdx.y * 14 + threadIdx.y;

    if(j >= width || i >= height)
    {
        return;
    }

    __shared__ int ccpy[16][16];
    ccpy[threadIdx.x][threadIdx.y] = inputVec[width * i + j];
    __syncthreads();	
    // Only Inner Part Need Compute Edges
    if(threadIdx.x == 0 || threadIdx.x == 15 || threadIdx.y == 0 || threadIdx.y == 15)
    {
        return;
    }

    int topLeft = ccpy[threadIdx.x - 1][threadIdx.y - 1];
    int topRight = ccpy[threadIdx.x + 1][threadIdx.y - 1];
    int bottomLeft =  ccpy[threadIdx.x - 1][threadIdx.y + 1];
    int bottomRight =  ccpy[threadIdx.x + 1][threadIdx.y + 1];

    int sum1 = topRight
        - topLeft
        + 2 *  ccpy[threadIdx.x + 1][threadIdx.y]
        - 2 *  ccpy[threadIdx.x - 1][threadIdx.y]
        + bottomRight
        - bottomLeft;
    int sum2 = topLeft
        + 2 *  ccpy[threadIdx.x][threadIdx.y - 1]
        + topRight
        - bottomLeft
        - 2 *  ccpy[threadIdx.x][threadIdx.y + 1]
        - bottomRight;
    
    if ((sum1 * sum1 + sum2 * sum2) > DEFAULT_THRESHOLD)
    {
        outVec[i * width + j] = 255;
    }
}

int main(int argc, char** argv)
{

    int thresh = DEFAULT_THRESHOLD;
    char* filename;
    filename = strdup(DEFAULT_FILENAME);

    if (argc > 1)
    {
        if (argc == 3)
        { // filename AND threshold
            filename = strdup(argv[1]);
            thresh = atoi(argv[2]);
        }
        if (argc == 2)
        { // default file but specified threshhold

            thresh = atoi(argv[1]);
        }

        fprintf(stderr, "file %s    threshold %d\n", filename, thresh);
    }

    int xsize, ysize, maxval;
    unsigned int* pic = read_ppm(filename, xsize, ysize, maxval);

    int numbytes = xsize * ysize * sizeof(int);
    int* result = (int*)malloc(numbytes);
    if (!result)
    {
        fprintf(stderr, "sobel() unable to malloc %d bytes\n", numbytes);
        exit(-1); // fail
    }

    cudaError_t cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) 
    {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
    }

    unsigned int* dev_in = 0;
    unsigned int* dev_out = 0;
    // Allocate GPU buffers for two vectors (one input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_in, ysize * xsize * sizeof(int));
    if (cudaStatus != cudaSuccess) 
    {
        fprintf(stderr, "cudaMalloc failed!");
    }

    cudaStatus = cudaMalloc((void**)&dev_out, ysize * xsize * sizeof(int));
    if (cudaStatus != cudaSuccess) 
    {
        fprintf(stderr, "cudaMalloc failed!");
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_in, pic, ysize * xsize * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) 
    {
        fprintf(stderr, "cudaMemcpy failed!");
    }
    cudaEvent_t start, stop;
    float elapsedTime;

    cudaEventCreate(&start);
    cudaEventRecord(start, 0);
    //// Kernel invocation
    // dim3 threadsPerBlock(16, 16);
    // dim3 numBlocks((xsize + 15) / threadsPerBlock.x, (ysize + 15) / threadsPerBlock.y);
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((xsize - 2 + 15) / 14, (ysize - 2 + 15) / 14);
    //MatAdd << <numBlocks, threadsPerBlock >> > (A, B, C);
    sobel_kernel_shared_mem_reuse << <numBlocks, threadsPerBlock >> > (dev_in, dev_out, xsize, ysize);
    cudaEventCreate(&stop);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Elapsed time : %f ms\n", elapsedTime);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) 
    {
        fprintf(stderr, "sobel_kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) 
    {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
    }

    // Copy output vectors from GPU buffers to host memory .
    cudaStatus = cudaMemcpy(result, dev_out, ysize * xsize * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy failed!");
    }

    write_ppm("result.ppm", xsize, ysize, 255, result);

    fprintf(stderr, "cuda sobel done\n");

    cudaFree(dev_in);
    cudaFree(dev_out);
    free(result);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) 
    {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

unsigned int* read_ppm(char* filename, int& xsize, int& ysize, int& maxval) {

    if (!filename || filename[0] == '\0') {
        fprintf(stderr, "read_ppm but no file name\n");
        return NULL;  // fail
    }

    fprintf(stderr, "read_ppm( %s )\n", filename);
    int fd = open(filename, O_RDONLY);
    if (fd == -1)
    {
        fprintf(stderr, "read_ppm()    ERROR  file '%s' cannot be opened for reading\n", filename);
        return NULL; // fail 

    }

    char chars[1024];
    int num = read(fd, chars, 1000);

    if (chars[0] != 'P' || chars[1] != '6')
    {
        fprintf(stderr, "Texture::Texture()    ERROR  file '%s' does not start with \"P6\"  I am expecting a binary PPM file\n", filename);
        return NULL;
    }

    unsigned int width, height, maxvalue;


    char* ptr = chars + 3; // P 6 newline
    if (*ptr == '#') // comment line! 
    {
        ptr = 1 + strstr(ptr, "\n");
    }

    num = sscanf(ptr, "%d\n%d\n%d", &width, &height, &maxvalue);
    fprintf(stderr, "read %d things   width %d  height %d  maxval %d\n", num, width, height, maxvalue);
    xsize = width;
    ysize = height;
    maxval = maxvalue;

    unsigned int* pic = (unsigned int*)malloc(width * height * sizeof(unsigned int));
    if (!pic) {
        fprintf(stderr, "read_ppm()  unable to allocate %d x %d unsigned ints for the picture\n", width, height);
        return NULL; // fail but return
    }

    // allocate buffer to read the rest of the file into
    int bufsize = 3 * width * height * sizeof(unsigned char);
    if (maxval > 255) bufsize *= 2;
    unsigned char* buf = (unsigned char*)malloc(bufsize);
    if (!buf) {
        fprintf(stderr, "read_ppm()  unable to allocate %d bytes of read buffer\n", bufsize);
        return NULL; // fail but return
    }





    // TODO really read
    char duh[80];
    char* line = chars;

    // find the start of the pixel data.   no doubt stupid
    sprintf(duh, "%d\0", xsize);
    line = strstr(line, duh);
    //fprintf(stderr, "%s found at offset %d\n", duh, line-chars);
    line += strlen(duh) + 1;

    sprintf(duh, "%d\0", ysize);
    line = strstr(line, duh);
    //fprintf(stderr, "%s found at offset %d\n", duh, line-chars);
    line += strlen(duh) + 1;

    sprintf(duh, "%d\0", maxval);
    line = strstr(line, duh);


    fprintf(stderr, "%s found at offset %d\n", duh, line - chars);
    line += strlen(duh) + 1;

    long offset = line - chars;
    lseek(fd, offset, SEEK_SET); // move to the correct offset
    long numread = read(fd, buf, bufsize);
    fprintf(stderr, "Texture %s   read %ld of %ld bytes\n", filename, numread, bufsize);

    close(fd);


    int pixels = xsize * ysize;
    for (int i = 0; i < pixels; i++) pic[i] = (int)buf[3 * i];  // red channel



    return pic; // success
}

void write_ppm(char* filename, int xsize, int ysize, int maxval, int* pic)
{
    FILE* fp;

    fp = fopen(filename, "w");
    if (!fp)
    {
        fprintf(stderr, "FAILED TO OPEN FILE '%s' for writing\n");
        exit(-1);
    }

    fprintf(fp, "P6\n");
    fprintf(fp, "%d %d\n%d\n", xsize, ysize, maxval);

    int numpix = xsize * ysize;
    for (int i = 0; i < numpix; i++) {
        unsigned char uc = (unsigned char)pic[i];
        fprintf(fp, "%c%c%c", uc, uc, uc);
    }
    fclose(fp);

}
