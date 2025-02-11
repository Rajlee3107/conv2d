#include <iostream>
#include <cuda.h>
#include <random>
#include <fstream>
#include <iomanip>

void readInput(const char* fileName, float*& h_input, int& H, int& W) {
    std::ifstream inFile(fileName);
    if (!inFile) {
        std::cerr << "Error: Could not open Input file " << fileName << std::endl;
        exit(EXIT_FAILURE);
    }

    inFile >> H >> W; // Using the extraction operator
    int size = H * W * sizeof(float);
    // Allocate the host input matrix
    h_input = (float *)malloc(size);

    for(int i = 0; i < H * W; i++) {
      if(!(inFile >> h_input[i])) {
	std::cerr << "Error: Invalid file format. Expected " << H * W << " elements." << std::endl;
        exit(EXIT_FAILURE); 
      }
    }
  }

void readFilter(const char* fileName, float*& h_filter, int& R) {
  std::ifstream inFile(fileName);
  if(!inFile) {
    std::cerr << "Error: Could not open Filter file " << fileName << std::endl;
    exit(EXIT_FAILURE);
  }

  inFile >> R;
  int size = R * R * sizeof(float);
  // Allocate the host filter matrix
  h_filter = (float *)malloc(size);

  for(int i = 0; i < R * R; i++) {
    if(!(inFile >> h_filter[i])) {
      std::cerr << "Error: Invalid file format. Expected " << R * R << " elements." << std::endl;
      exit(EXIT_FAILURE); 
    }
  }
}

void writeOutput(float* h_output, int H, int W) {
  
  for(int i = 0; i < H; i++) {
    for(int j = 0; j < W; j++) {
      std::cout << std::fixed << std::setprecision(3) << h_output[i * W + j] << std::endl;
    }
  }
}


//------- The cuda kernel -------//

// Both the input and the filter will have float values in them.
__global__ void conv2d_kernel(float* input, float* filter, float* output, int H, int W, int R) { 
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < H && col < W) {
    float sum = 0.0f;
    int pad = R/2;
    for(int i = -pad; i <= pad; i++) {
      for(int j = -pad; j <= pad; j++) {
        int r = row + i;
        int c = col + j;
        if(r >= 0 && r < H && c >= 0 && c < H) {
          int filter_row = i +pad;
          int filter_col = j + pad;
          sum += input[r * W + c] * filter[filter_row * R + filter_col];
        }
      }
    }
    output[row * W + col] = sum;
  }
}


int main(int argc, char *argv[]) {

  if(argc < 3) {
    std::cerr << "ERROR: number of arguments < 3" << std::endl;
    return -1;
  }

  // Error code to check return values for CUDA calls
  cudaError_t err = cudaSuccess;

  // ---- Read the inputs from command line ---- //
  int H, W, R;
  float *h_input, *h_filter;
  readInput(argv[1], h_input, H, W);
  readFilter(argv[2], h_filter, R);
  printf("[Vector multiplication of %0d x %0d by %0d x %0d matrices]\n", H, W, R, R);

  // ---- Allocate/move data using cudaMalloc and cudaMemCpy ---- //
  // Verify that allocations succeeded
  if (h_input == NULL || h_filter == NULL)
  {
    fprintf(stderr, "Failed to allocate host input and filter matrices!\n");
    exit(EXIT_FAILURE);
  }

  // Allocate the device input matrix
    float *d_input = NULL;
    err = cudaMalloc((void **)&d_input, H * W * sizeof(float));

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device matrix input (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device input vector B
    float *d_filter = NULL;
    err = cudaMalloc((void **)&d_filter, R * R * sizeof(float));

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device matrix filter (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device output vector C
    float *d_output = NULL;
    err = cudaMalloc((void **)&d_output, H * W * sizeof(float));

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device matrix output (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the host input matrices input and filter in host memory to the device input and filter matrices in
    // device memory
    printf("Copy input data from the host memory to the CUDA device\n");
    err = cudaMemcpy(d_input, h_input, H * W * sizeof(float), cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy matrix input from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_filter, h_filter, R * R * sizeof(float), cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy matrix filter from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

  // ---- Launch the kernel ---- //
  // Define grid and block dimensions
  dim3 blockDim(16, 16);
  dim3 gridDim((W + 15) / 16, (H + 15) / 16);

  printf("CUDA kernel launch with %d blocks of %d threads\n", gridDim, blockDim);
  conv2d_kernel<<<gridDim, blockDim>>>(d_input, d_filter, d_output, H, W, R);
  err = cudaGetLastError();

  if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch 2D convolution kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

  // ----- Copy results back to host ----- //
  // Allocate the host output matrix
  float *h_output = (float *)malloc(H * W * sizeof(float));

  // Verify that allocations succeeded
  if (h_output == NULL)
  {
    fprintf(stderr, "Failed to allocate host output matrix!\n");
    exit(EXIT_FAILURE);
  }

  printf("Copy output matrix from the CUDA device to the host memory\n");
  err = cudaMemcpy(h_output, d_output, H * W * sizeof(float), cudaMemcpyDeviceToHost);

  if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy output matrix from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

  // Print the output
  writeOutput(h_output, H, W);
  printf("Test Completed\n");

  // ---- Clean up the memory ----- //
  err = cudaFree(d_input);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device input matrix (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_filter);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device filter matrix (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_output);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device output matrix (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Free host memory
    free(h_input);
    free(h_filter);
    free(h_output);

    // Reset the device and exit
    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    err = cudaDeviceReset();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    printf("Done\n");


  return 0;
}
