#include "cpuLib.h"
#include "cudaLib.cuh"
#include <cuda_runtime.h>

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort)
	{
	if (code != cudaSuccess)
	{
	fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
	if (abort) exit(code);
	}
}

__global__
void saxpy_gpu (float* x, float* y, float scale, int size) {
	// Insert GPU SAXPY kernel code here
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if(i>=size) return;
	if(i<size) y[i] = scale*x[i] + y[i];
}

int runGpuSaxpy(int vectorSize) {


	// Insert code here
	float *a, *b, *c, *d_x, *d_y;
	a = (float*)malloc(vectorSize*sizeof(float));
	b = (float*)malloc(vectorSize*sizeof(float));
	c = (float*)malloc(vectorSize*sizeof(float));
	cudaMalloc(&d_x, vectorSize*sizeof(float));
	cudaMalloc(&d_y, vectorSize*sizeof(float));
	vectorInit(a, vectorSize);
	vectorInit(b, vectorSize);
	cudaMemcpy(d_x, a, vectorSize*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, b, vectorSize*sizeof(float), cudaMemcpyHostToDevice);
	int device = 0;
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, device);
	int maxThreadPerBloc = prop.maxThreadsPerBlock;
	int blockSize = maxThreadPerBloc;
	blockSize -= (blockSize%32);
	int gridSize = (vectorSize+blockSize-1)/blockSize;


	saxpy_gpu<<<gridSize, blockSize>>>(d_x, d_y, 2.0f, vectorSize);
	cudaMemcpy(c, d_y, vectorSize*sizeof(float),cudaMemcpyDeviceToHost);
	cudaFree(d_x);
	cudaFree(d_y);

	int errorCount = verifyVector(a, b, c, 2.0f, vectorSize);
	std::cout << "Found" <<errorCount << "/" << vectorSize << " errors \n";
	free(a);
	free(b);
	free(c);
	return 0;
}

/*
 Some helpful definitions

 generateThreadCount is the number of threads spawned initially. Each thread is responsible for sampleSize points.
 *pSums is a pointer to an array that holds the number of 'hit' points for each thread. The length of this array is pSumSize.

 reduceThreadCount is the number of threads used to reduce the partial sums.
 *totals is a pointer to an array that holds reduced values.
 reduceSize is the number of partial sums that each reduceThreadCount reduces.

*/

__global__
void generatePoints (uint64_t * pSums, uint64_t pSumSize, uint64_t sampleSize) {
	// Insert code here
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(pSumSize <= idx) return;
	curandState_t rng;
	curand_init(clock64(), idx, 0, &rng);

	uint64_t hits = 0;
	for(uint64_t i=0; i< sampleSize; ++ i){

	float x = curand_uniform(&rng);
	float y = curand_uniform(&rng);

	if(int(x*x + y*y) == 0){
	++ hits;
	}
	}
	pSums[idx] = hits;
}

__global__
void reduceCounts (uint64_t * pSums, uint64_t * totals, uint64_t pSumSize, uint64_t reduceSize) {
	// Insert code here
	uint64_t threadId = blockIdx.x * blockDim.x + threadIdx.x;
	uint64_t start = threadId * reduceSize;
	if(reduceSize <= threadId) return;
	uint64_t sum = 0;
	for(uint64_t i = 0; i < reduceSize; ++ i){
	if((threadId*reduceSize + i)>=pSumSize) return;
	sum += pSums[start + i];
	}
	totals[threadId] = sum;
}

int runGpuMCPi (uint64_t generateThreadCount, uint64_t sampleSize,
	uint64_t reduceThreadCount, uint64_t reduceSize) {

	//  Check CUDA device presence
	int numDev;
	cudaGetDeviceCount(&numDev);
	if (numDev < 1) {
	std::cout << "CUDA device missing!\n";
	return -1;
	}

	auto tStart = std::chrono::high_resolution_clock::now();

	//float approxPi = estimatePi(4, 3,
	// 2, 2);
	float approxPi = estimatePi(generateThreadCount, sampleSize,
	reduceThreadCount, reduceSize);

	std::cout << "Estimated Pi = " << approxPi << "\n";

	auto tEnd= std::chrono::high_resolution_clock::now();

	std::chrono::duration<double> time_span = (tEnd- tStart);
	std::cout << "It took " << time_span.count() << " seconds.";

	return 0;
}

double estimatePi(uint64_t generateThreadCount, uint64_t sampleSize,
	uint64_t reduceThreadCount, uint64_t reduceSize) {

	double approxPi = 0;
	uint64_t *d_pSums, *d_totals;
	cudaMalloc(&d_pSums, sizeof(uint64_t)*generateThreadCount);
	cudaMalloc(&d_totals, sizeof(uint64_t)*reduceThreadCount);
	cudaDeviceProp prop;
	int device = 0;
	cudaGetDeviceProperties(&prop, device);
	int blockSize = prop.maxThreadsPerBlock;
	blockSize -= (blockSize%32);
	int gridSize = (generateThreadCount + blockSize -1)/blockSize;
	generatePoints<<<gridSize, blockSize>>>(d_pSums, generateThreadCount, sampleSize);
	cudaDeviceSynchronize();
	gridSize = (reduceThreadCount + blockSize - 1)/blockSize;
	reduceCounts<<<gridSize, blockSize>>>(d_pSums, d_totals, generateThreadCount, reduceSize);
	uint64_t *totals = new uint64_t[reduceThreadCount];
	cudaMemcpy(totals, d_totals, sizeof(uint64_t) * reduceThreadCount, cudaMemcpyDeviceToHost);
	uint64_t grandTotal = 0;
	for(uint64_t i=0; i<reduceThreadCount; ++ i){
		grandTotal += totals[i];
	}
	approxPi = 4.0 * grandTotal/((double)generateThreadCount * (double)sampleSize);
	delete [] totals;
	cudaFree(d_pSums);
	cudaFree(d_totals);
	return approxPi;
}
