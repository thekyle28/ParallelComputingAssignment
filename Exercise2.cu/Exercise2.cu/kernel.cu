/**
* Name: Kyle Allen-Taylor
* SID: 1410908
* Achieved: All parts (non extended)
*
* Results:
* windowSize = 100, numberElements= 10,000,000
* time parallel:   0.06861s
* time sequential: 0.07354s
* speedup: 1.07186
*
* windowSize = 100, numberElements= 1,000,000
* time parallel:   0.00794s
* time sequential: 0.01595s
* speedup: 2.00881
*
* windowSize = 100, numberElements= 100,000
* time parallel:   0.00095s
* time sequential: 0.00279s
* speedup: 2.93684
*/

#include <stdio.h>
#include <time.h>
#include <stdbool.h>
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
#define BLOCK_SIZE 256 // the actual configured block size
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
__global__ void blockScan(unsigned int *X, unsigned int *Y, unsigned int len, unsigned int *extractedSum, bool extractSum) {
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
	unsigned int index = (blockIdx.x + 1) * blockDim.x + threadIdx.x;
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
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
	int subIndex = index - windowSize;
	if (index < numElements) {
		if (subIndex < 0)
			output[index] = (float)scannedVector[index] / windowSize;
		else
			output[index] = (float)(scannedVector[index] - scannedVector[subIndex]) / windowSize;
	}
}

/**
* Initialise vectors on the device
*/
__host__ void checkErr(cudaError_t err, char *errorMessage) {
	if (err != cudaSuccess)
	{
		fprintf(stderr, "%s (error code %s)!\n", errorMessage, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
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
	unsigned int numElements = 100000;
	int windowSize = 100;
	// Define number of threads per block
	int threadsPerBlock = BLOCK_SIZE;
#ifdef DEBUG
	unsigned int init_h_A[] = { 1,2,1,3,1,1,3,3,2,1,2,2,2,1,1,2 };
	numElements = 16;
#endif
	size_t size = numElements * sizeof(int);
	size_t floatSize = numElements * sizeof(float);
	printf("[Calculating windowed average of %d elements]\n", numElements);

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
	checkErr(err, "Failed to allocate device vector A");


	// Allocate the device output vector B for the scan
	unsigned int *d_B = NULL;
	err = cudaMalloc((void **)&d_B, size);
	checkErr(err, "Failed to allocate device vector B");

	// Allocate the device output vector d_windowedAverage for the scan
	float *d_windowedAverage = NULL;
	err = cudaMalloc((void **)&d_windowedAverage, floatSize);
	checkErr(err, "Failed to allocate device vector windowedAverage");

	// Define the size of the Sum1 vectors to store the extracted sum
	unsigned int sum1NumElems = ceil(numElements / threadsPerBlock);
	unsigned int sum1Size = sum1NumElems * sizeof(int);
	// Allocate the device vector d_Sum1 to store the extracted sum of the first level scan
	unsigned int *d_Sum1 = NULL;
	err = cudaMalloc((void **)&d_Sum1, sum1Size);
	checkErr(err, "Failed to allocate device vector Sum1");

	// Allocate the device vector d_Sum1_scanned to store the scan of the extracted first level
	unsigned int *d_Sum1_scanned = NULL;
	err = cudaMalloc((void **)&d_Sum1_scanned, sum1Size);
	checkErr(err, "Failed to allocate device vector Sum1_scanned");

	// Define the size of the Sum2 vectors to store the extracted sum
	unsigned int sum2NumElems = ceil(sum1NumElems / threadsPerBlock);
	unsigned int sum2Size = sum2NumElems * sizeof(int);
	// Allocate the device vector d_Sum1 to store the extracted sum of the first level scan
	unsigned int *d_Sum2 = NULL;
	err = cudaMalloc((void **)&d_Sum2, sum2Size);
	checkErr(err, "Failed to allocate device vector Sum2");

	unsigned int *d_Sum2_scanned = NULL;
	err = cudaMalloc((void **)&d_Sum2_scanned, sum2Size);
	checkErr(err, "Failed to allocate device vector Sum2_scanned");

	// Copy the host input vector A into host memory to the device input vector in
	// device memory
	printf("Copy input data from the host memory to the CUDA device\n");
	err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
	checkErr(err, "Failed to copy vector A from host to device");

#ifdef DEBUG
	threadsPerBlock = 4;
#endif
	int blocksPerGrid = 1 + ((numElements - 1) / threadsPerBlock);
	int blocksPerGridSum1 = 1 + ((sum1NumElems - 1) / threadsPerBlock);
	int blocksPerGridSum2 = 1 + ((sum2NumElems - 1) / threadsPerBlock);

	printf("Launch the CUDA kernel with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);


#ifdef CUDA_TIMING
	cudaEvent_t start, stop;
	float time;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start, 0);
#endif
#ifdef DEBUG
	blockScan << <blocksPerGrid, threadsPerBlock >> >(d_Sum1, d_Sum1_scanned, sum1NumElems, d_Sum1, false);
	blockAdd << <blocksPerGrid, threadsPerBlock >> >(d_Sum1_scanned, d_B, sum1NumElems, numElements);
	d_WindowedAverage << <blocksPerGrid, threadsPerBlock >> >(d_B, h_C, windowSize, numElements);
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
	blockScan << <blocksPerGrid, threadsPerBlock >> >(d_A, d_B, numElements, d_Sum1, true);
	blockScan << <blocksPerGridSum1, threadsPerBlock >> >(d_Sum1, d_Sum1_scanned, sum1NumElems, d_Sum2, true);
	blockScan << <blocksPerGridSum2, threadsPerBlock >> >(d_Sum2, d_Sum2_scanned, sum2NumElems, d_Sum1, false);
	blockAdd << <blocksPerGridSum1, threadsPerBlock >> >(d_Sum2_scanned, d_Sum1_scanned, sum2NumElems, sum1NumElems);
	blockAdd << <blocksPerGrid, threadsPerBlock >> >(d_Sum1_scanned, d_B, sum1NumElems, numElements);
	d_CalcWindowedAverage << <blocksPerGrid, threadsPerBlock >> >(d_B, d_windowedAverage, windowSize, numElements);
#endif


	// Stop the device timer
#ifdef CUDA_TIMING
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	err = cudaEventElapsedTime(&time, start, stop);
	checkErr(err, "Failed to get elapsed time ");

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	printf("CUDA_TIMING: %.5f s\n", time / 1000.0);
#endif

	// Wait for device to finish
	cudaDeviceSynchronize();


	// Copy the device result vector in device memory to the host result vector
	// in host memory.
	printf("Copy output data from the CUDA device to the host memory\n");
	err = cudaMemcpy(h_windowedAverage, d_windowedAverage, floatSize, cudaMemcpyDeviceToHost);
	checkErr(err, "Failed to copy vector windowedAverage from device to host");

	// Verify that the result vector is correct

	// Perform the timed sequential version of the windowed average
#ifdef TIMING_SUPPORT
	StopWatchInterface *timer = NULL;
	sdkCreateTimer(&timer);             // create a timer
	sdkStartTimer(&timer);               // start the timer
#endif
	sequentialWindowedAverage(h_A, seqWindowedAverage, windowSize, numElements);
#ifdef TIMING_SUPPORT
	// stop and destroy timer
	sdkStopTimer(&timer);
	double dSeconds = sdkGetTimerValue(&timer) / (1000.0);

	printf("Time = %.5f s\n", dSeconds);
	sdkDeleteTimer(&timer);
#endif

#ifdef DEBUG
	printVector(seqWindowedAverage, numElements, "windowedAverage");
#endif
	for (int i = 0; i < numElements; ++i)
	{
		if (seqWindowedAverage[i] != h_windowedAverage[i])
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
	checkErr(err, "Failed to free device vector A");

	err = cudaFree(d_B);
	checkErr(err, "Failed to free device vector B");

	err = cudaFree(d_windowedAverage);
	checkErr(err, "Failed to free device vector windowedAverage");

	err = cudaFree(d_Sum1);
	checkErr(err, "Failed to free device vector Sum1");

	err = cudaFree(d_Sum1_scanned);
	checkErr(err, "Failed to free device vector Sum1_scanned");

	err = cudaFree(d_Sum2);
	checkErr(err, "Failed to free device vector Sum2");

	err = cudaFree(d_Sum2_scanned);
	checkErr(err, "Failed to free device vector Sum2_scanned");

	// Free host memory
	free(h_A);
	free(h_B);
	free(h_windowedAverage);
	free(seqWindowedAverage);

	// Reset the device and exit
	err = cudaDeviceReset();
	checkErr(err, "Failed to reset the device");

	printf("Done\n");
	return 0;
}
