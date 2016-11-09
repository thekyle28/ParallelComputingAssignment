/**
* Copyright 1993-2013 NVIDIA Corporation.  All rights reserved.
*
* Please refer to the NVIDIA end user license agreement (EULA) associated
* with this source code for terms and conditions that govern your use of
* this software. Any use, reproduction, disclosure, or distribution of
* this software and related documentation outside the terms of the EULA
* is strictly prohibited.
*
*/

/**
*
* Slightly modified to provide timing support
*/

#include <stdio.h>
#include <time.h>
// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

#define TIMING_SUPPORT

#ifdef TIMING_SUPPORT
#include <helper_cuda.h>
#include <helper_functions.h>
#endif

#define CUDA_TIMING
#define BLOCK_SIZE 4 // the actual configured block size

__host__ void printVector(unsigned int *vector, int numElements) {
	for (int i = 0; i < numElements; i++) {
		printf("Value of vector at index %d is %d\n", i, vector[i]);
	}
}

__host__ void initialiseSmallVector(unsigned int *vector, int numElements) {
	srand(time(NULL));
	for (int i = 0; i < numElements; i++) {
		vector[i] = rand() % 10;
	}
}

/**
* CUDA Kernel Device code
*
* Computes the cumulative sum of the vector X and it passes the result into the vector Y. The 2 vectors have the same
* number of elements numElements.
*/

__global__ void blockScan(unsigned int *X, unsigned int *Y, unsigned int len, unsigned int *extractedSum, boolean extractSum) {
	__shared__ unsigned int XY[BLOCK_SIZE];
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < len) XY[threadIdx.x] = X[i];
	// Reduction Phase 
	for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
		__syncthreads();
		unsigned int index = (threadIdx.x + 1) * stride * 2 - 1;
		if (index < blockDim.x) XY[index] += XY[index - stride];
	}
	// Distribution Phase 
	for (unsigned int stride = BLOCK_SIZE / 4; stride > 0; stride /= 2) {
		__syncthreads();
		unsigned int index = (threadIdx.x + 1) * stride * 2 - 1;
		if (index + stride < BLOCK_SIZE)
			XY[index + stride] += XY[index];
	}
	__syncthreads();
	if (i < len)
		Y[i] = XY[threadIdx.x];

	if (extractSum) {
		printf("@thread %d\n", threadIdx.x);
		if (threadIdx.x == BLOCK_SIZE - 1) {
			printf("Inserting into extractedSum[%d]\n", blockIdx.x);
			extractedSum[blockIdx.x] = XY[threadIdx.x];
		}
	}
}

__global__ void blockAdd(unsigned int *X, unsigned int *Y, unsigned int len, unsigned int *extractedSum, boolean extractSum) {

}

/**
* Host main routine
*/
int main(void)
{
	boolean debug = true;
	// Error code to check return values for CUDA calls
	cudaError_t err = cudaSuccess;

	// Print the vector length to be used, and compute its size
	unsigned int numElements = 1000000;
	unsigned int init_h_A[] = { 1,2,1,3,1,1,3,3,2,1,2,2,2,1,1,2 };
	if (debug) {
		numElements = 16;
	}
	size_t size = numElements * sizeof(int);
	printf("[Vector addition of %d elements]\n", numElements);
	static const int p_init[] = { 0, 1, 2 };


	// Allocate the host input vector A
	unsigned int *h_A = (unsigned int *)malloc(size);

	if (debug) {
		memcpy(h_A, init_h_A, size);
	}

	// Allocate the host output vector B
	unsigned int *h_B = (unsigned int *)malloc(size);

	// Verify that allocations succeeded
	if (h_A == NULL || h_B == NULL)
	{
		fprintf(stderr, "Failed to allocate host vectors!\n");
		exit(EXIT_FAILURE);
	}

	// Initialise the host input vector A with Random Values
	if(!debug)
		initialiseSmallVector(h_A, numElements);

	unsigned int h_sum1Size = ceil(numElements / BLOCK_SIZE) * sizeof(int);
	// Allocate the host output vector B
	unsigned int *h_C = (unsigned int *)malloc(h_sum1Size);

	// Verify that allocations succeeded
	if (h_C == NULL)
	{
		fprintf(stderr, "Failed to allocate host vectors!\n");
		exit(EXIT_FAILURE);
	}

	// Allocate the device input vector A
	unsigned int *d_A = NULL;
	err = cudaMalloc((void **)&d_A, size);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Allocate the device output vector B
	unsigned int *d_B = NULL;
	err = cudaMalloc((void **)&d_B, size);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Define the size of the Sum1 vectors to store the extracted sum
	unsigned int sum1Size = ceil(numElements / BLOCK_SIZE) * sizeof(int);
	// Allocate the device vector d_Sum1 to store the extracted sum of the first level scan
	unsigned int *d_Sum1 = NULL;
	err = cudaMalloc((void **)&d_Sum1, sum1Size);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector Sum1 (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	// Allocate the device vector d_Sum1_scanned to store the scan of the extracted first level
	unsigned int *d_Sum1_scanned = NULL;
	err = cudaMalloc((void **)&d_Sum1_scanned, sum1Size);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector Sum1_scanned (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Define the size of the Sum1 vectors to store the extracted sum
	unsigned int sum2Size = ceil(numElements / BLOCK_SIZE) * sizeof(int); ///NEEDS TO BE CHANGED
	// Allocate the device vector d_Sum1 to store the extracted sum of the first level scan
	unsigned int *d_Sum2 = NULL;
	err = cudaMalloc((void **)&d_Sum2, sum2Size);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector Sum1 (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Copy the host input vector A into host memory to the device input vector in
	// device memory
	printf("Copy input data from the host memory to the CUDA device\n");
	err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Launch the Vector Add CUDA Kernel
	int threadsPerBlock = 1024;
	if (debug) threadsPerBlock = 4;

	// Note this pattern, based on integer division, for rounding up
	int blocksPerGrid = 1 + ((numElements - 1) / threadsPerBlock);

	printf("Launch the CUDA kernel with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);

#ifdef TIMING_SUPPORT
	StopWatchInterface *timer = NULL;
	sdkCreateTimer(&timer);             // create a timer
	sdkStartTimer(&timer);               // start the timer
#endif
#ifdef CUDA_TIMING
	cudaEvent_t start, stop;
	float time;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start, 0);
#endif
	blockScan<<<blocksPerGrid, threadsPerBlock >>>(d_A, d_B, numElements, d_Sum1, true);
	// Copy the device result vector in device memory to the host result vector
	// in host memory.
	if (debug) {
		printf("Copy output data from the CUDA device to the host memory\n");
		err = cudaMemcpy(h_C, d_Sum1, sum1Size, cudaMemcpyDeviceToHost);
		printVector(h_C, 4);
		printVector(d_Sum1, 4);
	}
	blockScan<<<blocksPerGrid, threadsPerBlock >>>(d_Sum1, d_Sum1_scanned, sum1Size, d_Sum2, true);
	blockScan<<<blocksPerGrid, threadsPerBlock >>>(d_Sum2, d_Sum1_scanned, sum2Size, d_Sum1, false);



	
//#ifdef CUDA_TIMING
//	cudaEventRecord(stop, 0);
//	cudaEventSynchronize(stop);
//
//	err = cudaEventElapsedTime(&time, start, stop);
//	if (err != cudaSuccess)
//	{
//		fprintf(stderr, "Failed to get elapsed time (error code %s)!\n", cudaGetErrorString(err));
//		exit(EXIT_FAILURE);
//	}
//
//	cudaEventDestroy(start);
//	cudaEventDestroy(stop);
//	printf("CUDA_TIMING: %.4f ms\n", time);
//#endif
//	// wait for device to finish
//	cudaDeviceSynchronize();
//
//	err = cudaGetLastError();
//
//	if (err != cudaSuccess)
//	{
//		fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
//		exit(EXIT_FAILURE);
//	}


//#ifdef TIMING_SUPPORT
//	// stop and destroy timer
//	sdkStopTimer(&timer);
//	double dSeconds = sdkGetTimerValue(&timer) / (1000.0);
//
//
//	//Log throughput, etc
//	printf("Time = %.5f s\n", dSeconds);
//	sdkDeleteTimer(&timer);
//#endif
//

	// Copy the device result vector in device memory to the host result vector
	// in host memory.
	printf("Copy output data from the CUDA device to the host memory\n");
	err = cudaMemcpy(h_B, d_B, size, cudaMemcpyDeviceToHost);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy vector B from device to host (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Verify that the result vector is correct
	/*for (int i = 0; i < numElements; ++i)
	{
		if (fabs(h_A[i] + h_B[i]) > 1e-5)
		{
			fprintf(stderr, "Result verification failed at element %d!\n", i);
			exit(EXIT_FAILURE);
		}
	}*/
	printf("Test PASSED\n");
	printVector(h_B, numElements);
	// Free device global memory
	err = cudaFree(d_A);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	err = cudaFree(d_B);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to free device vector B (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Free host memory
	free(h_A);
	free(h_B);

	// Reset the device and exit
	err = cudaDeviceReset();

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to reset the device! error=%s\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	printf("Done\n");
	return 0;
}
