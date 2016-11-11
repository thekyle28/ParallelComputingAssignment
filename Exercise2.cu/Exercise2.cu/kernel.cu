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
//#define DEBUG
#define CUDA_TIMING
#ifndef DEBUG
#define BLOCK_SIZE 1024 // the actual configured block size
#else
#define BLOCK_SIZE 4
#endif

/**
* Prints a float vector, with numElements the number of elements in the vector and vectorName, the vector variable name.
*/
__host__ void printVector(float *vector, int numElements, char *vectorName) {
	for (int i = 0; i < numElements; i++) {
		printf("Value of %s at index %d is %f\n", vectorName, i, vector[i]);
	}
}

/**
* Prints an int vector, with numElements the number of elements in the vector and vectorName, the vector variable name.
*/
__host__ void printVector(unsigned int *vector, int numElements, char *vectorName) {
	for (int i = 0; i < numElements; i++) {
		printf("Value of %s at index %d is %d\n", vectorName, i, vector[i]);
	}
}

/**
* Initialise a given vector with random numbers between 0 and 9 where numElements is the number of elements 
* in the vector.
*/
__host__ void initialiseSmallVector(unsigned int *vector, int numElements) {
	srand(time(NULL));
	for (int i = 0; i < numElements; i++) {
		vector[i] = rand() % 10;
	}
}

/**
* Computes the sequential scan of a vector.
*/
__host__ unsigned int *sequentialScan(unsigned int *vector, unsigned int *output, int numElements) {
	int cumulativeSum = 0;
	for (int i = 0; i < numElements; i++) {
		cumulativeSum += vector[i];
		output[i] = cumulativeSum;
	}
	return output;
}

/**
* Computes the sequential windowed Average with a given scanned vector.
*/
__host__ float *calcScannedWindowedAverage(unsigned int *scannedVector, float *output, int windowSize, int numElements) {
	int subIndex;
	for (int i = 0; i < numElements; i++) {
		subIndex = i - windowSize;
		if (subIndex < 0) 
			output[i] = (float)scannedVector[i] / windowSize;
		else
			output[i] = (float)(scannedVector[i] - scannedVector[subIndex]) / windowSize;
	}
	return output;
}
/**
* Computes the Windowed average of a given vector sequentially on the host. vector is the vector that the windowed
* average will be calculated on. output is the vector to store the result of the windowed average.
* windowSize is the size of the window for the average and numElements is the number of elements in both vectors.
*/
__host__ float *sequentialWindowedAverage(unsigned int *vector, float *output, int windowSize, int numElements) {
	unsigned int *scannedVector = (unsigned int*)malloc(numElements * sizeof(int));
	scannedVector = sequentialScan(vector, scannedVector, numElements);
	output = calcScannedWindowedAverage(scannedVector, output, windowSize, numElements);
	return output;
}

/**
* Computes the cumulative sum of the vector X and it passes the result into the vector Y. The 2 vectors have the same
* number of elements numElements. extractedSum is the vector where the values at the end of each block will be placed.
* extractSum is a boolean to say if we wish to extract the values at the end of each block or not.
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
		if (threadIdx.x == BLOCK_SIZE - 1) {
			extractedSum[blockIdx.x] = XY[threadIdx.x];
		}
	}
}

/**
* Adds the extract vector to the output vector. extractLen is the length of the extract vector and outputLen is the 
* length of the output vector.
*/
__global__ void blockAdd(unsigned int *extract, unsigned int *output, unsigned int extractLen, unsigned int outputLen) {
	unsigned int index = (blockIdx.x + 1) * BLOCK_SIZE + threadIdx.x;
	if (index < outputLen) {
		output[index] = output[index] + extract[blockIdx.x];
#ifdef DEBUG
		printf("inserted value %d at index %d\n", output[index], index);
#endif
	}
}
/**
* Calculates the windowed average of a scanned vector.
*/
__global__ void d_CalcWindowedAverage(unsigned int *scannedVector, float* output, int windowSize, int numElements) {
	unsigned int index = blockIdx.x * BLOCK_SIZE + threadIdx.x;
	int subIndex = index - windowSize;
	if (index < numElements) {
		if(subIndex < 0) 
			output[index] = (float) scannedVector[index] / windowSize;
		else
			output[index] = (float)(scannedVector[index] - scannedVector[subIndex]) / windowSize;
	}
}

/**
* Host main routine
*/
int main(void)
{
	// Error code to check return values for CUDA calls
	cudaError_t err = cudaSuccess;

	// Print the vector length to be used, and compute its size
	unsigned int numElements = 10000000;
	int windowSize = 4;
#ifdef DEBUG
	unsigned int init_h_A[] = { 1,2,1,3,1,1,3,3,2,1,2,2,2,1,1,2 };
	numElements = 16;
#endif
	size_t size = numElements * sizeof(int);
	size_t floatSize = numElements * sizeof(float);
	printf("[Vector addition of %d elements]\n", numElements);

	// Allocate the host input vector A
	unsigned int *h_A = (unsigned int *)malloc(size);

#ifdef DEBUG	
	memcpy(h_A, init_h_A, size);
#endif
	
	// Allocate the host output vector B
	unsigned int *h_B = (unsigned int *)malloc(size);

	// Verify that allocations succeeded
	if (h_A == NULL || h_B == NULL)
	{
		fprintf(stderr, "Failed to allocate host vectors!\n");
		exit(EXIT_FAILURE);
	}

	// Initialise the host input vector A with Random Values
#ifndef DEBUG		
	initialiseSmallVector(h_A, numElements);
#endif

	// Allocate the host output vector C for debugging
#ifdef DEBUG
	float *h_C = (float*)malloc(floatSize);
#endif
	// Initialise the host output vector for the windowed average of both the parallel and sequential versions
	float *h_windowedAverage = (float *)malloc(floatSize);
	float *seqWindowedAverage = (float *)malloc(floatSize);

#ifdef DEBUG

		// Verify that allocations succeeded
		if (h_C == NULL)
		{
			fprintf(stderr, "Failed to allocate host vectors!\n");
			exit(EXIT_FAILURE);
		}
#endif

	// Allocate the device input vector A
	unsigned int *d_A = NULL;
	err = cudaMalloc((void **)&d_A, size);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Allocate the device output vector B for the scan
	unsigned int *d_B = NULL;
	err = cudaMalloc((void **)&d_B, size);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Allocate the device output vector d_windowedAverage for the scan
	float *d_windowedAverage = NULL;
	err = cudaMalloc((void **)&d_windowedAverage, floatSize);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}



	// Define the size of the Sum1 vectors to store the extracted sum
	unsigned int sum1NumElems = ceil(numElements / BLOCK_SIZE);
	unsigned int sum1Size = sum1NumElems * sizeof(int);
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

	// Define the size of the Sum2 vectors to store the extracted sum
	unsigned int sum2NumElems =  ceil(sum1NumElems/ BLOCK_SIZE);
	unsigned int sum2Size = sum2NumElems * sizeof(int); 
	// Allocate the device vector d_Sum1 to store the extracted sum of the first level scan
	unsigned int *d_Sum2 = NULL;
	err = cudaMalloc((void **)&d_Sum2, sum2Size);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector Sum1 (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	unsigned int *d_Sum2_scanned = NULL;
	err = cudaMalloc((void **)&d_Sum2_scanned, sum2Size);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector Sum1_scanned (error code %s)!\n", cudaGetErrorString(err));
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

	// Define number of threads per block
	int threadsPerBlock = 1024;
#ifdef DEBUG
	threadsPerBlock = 4;
#endif
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
#ifdef DEBUG
	blockScan<<<blocksPerGrid, threadsPerBlock>>>(d_Sum1, d_Sum1_scanned, sum1NumElems, d_Sum1, false);
	blockAdd<<<blocksPerGrid, threadsPerBlock >>>(d_Sum1_scanned, d_B, sum1NumElems, numElements);
	d_WindowedAverage<<<blocksPerGrid, threadsPerBlock >>>(d_B, h_C, windowSize, numElements);
	err = cudaMemcpy(h_C, d_B, size, cudaMemcpyDeviceToHost);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to get elapsed time (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	printVector(h_C, numElements, "h_C");
	printf("Printed vector\n");
#endif
#ifndef DEBUG
	blockScan <<<blocksPerGrid, threadsPerBlock >>>(d_A, d_B, numElements, d_Sum1, true);
	blockScan<<<blocksPerGrid, threadsPerBlock >>>(d_Sum1, d_Sum1_scanned, sum1NumElems, d_Sum2, true);
	blockScan<<<blocksPerGrid, threadsPerBlock >>>(d_Sum2, d_Sum2_scanned, sum2NumElems, d_Sum1, false);
	blockAdd <<<blocksPerGrid, threadsPerBlock >>>(d_Sum2_scanned, d_Sum1_scanned, sum2NumElems, sum1NumElems);
	blockAdd <<<blocksPerGrid, threadsPerBlock >>>(d_Sum1_scanned, d_B, sum1NumElems, numElements);
	d_CalcWindowedAverage<<<blocksPerGrid, threadsPerBlock >>>(d_B, d_windowedAverage, windowSize, numElements);
#endif
	// Wait for device to finish
	cudaDeviceSynchronize();

	// Stop the device timer
#ifdef TIMING_SUPPORT
	// stop and destroy timer
	sdkStopTimer(&timer);
	double dSeconds = sdkGetTimerValue(&timer) / (1000.0);


	//Log throughput, etc
	printf("Time = %.5f s\n", dSeconds);
	sdkDeleteTimer(&timer);
#endif

	err = cudaGetLastError();

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}


#/*ifdef CUDA_TIMING
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	err = cudaEventElapsedTime(&time, start, stop);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to get elapsed time (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	printf("CUDA_TIMING: %.4f ms\n", time);
#endif	*/





	// Copy the device result vector in device memory to the host result vector
	// in host memory.
	printf("Copy output data from the CUDA device to the host memory\n");
	err = cudaMemcpy(h_windowedAverage, d_windowedAverage, floatSize, cudaMemcpyDeviceToHost);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy vector windowedAverage from device to host (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Verify that the result vector is correct

	// Perform the timed sequential version of the windowed average

	sequentialWindowedAverage(h_A, seqWindowedAverage, windowSize, numElements);
#ifdef DEBUG
	printVector(seqWindowedAverage, numElements, "windowedAverage");
#endif
	for (int i = 0; i < numElements; ++i)
	{
		if (seqWindowedAverage[i] != h_windowedAverage[i] )
		{
			fprintf(stderr, "Result verification failed at element %d!\n", i);
			exit(EXIT_FAILURE);
		}
	}
	printf("Test PASSED\n");
#ifdef DEBGUG 
	printVector(h_B, numElements, "h_B");
#endif
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
