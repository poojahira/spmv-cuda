#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <cuda.h>
#include <sys/time.h>
#include "mmio.h"

#define warpSize 32
#define BlockDim 1024
#define ITER 3

template <typename T>
__global__ void spmv_csr_vector_kernel(T * d_val,T * d_vector,int * d_cols,int * d_ptr,int N, T * d_out)
{
    	// Thread ID in block
    	int t = threadIdx.x;

    	// Thread ID in warp
    	int lane = t & (warpSize-1);

    	// Number of warps per block
    	int warpsPerBlock = blockDim.x / warpSize;

    	// One row per warp
    	int row = (blockIdx.x * warpsPerBlock) + (t / warpSize);

    	__shared__ volatile T vals[BlockDim];
    
    	if (row < N)
    	{
        	int rowStart = d_ptr[row];
        	int rowEnd = d_ptr[row+1];
        	T sum = 0;

        	// Use all threads in a warp accumulate multiplied elements
        	for (int j = rowStart + lane; j < rowEnd; j += warpSize)
        	{
            		int col = d_cols[j];
            		sum += d_val[j] * d_vector[col];
        	}
        	vals[t] = sum;
		__syncthreads();
        
        	// Reduce partial sums
        	if (lane < 16) vals[t] += vals[t + 16];
        	if (lane <  8) vals[t] += vals[t + 8];
        	if (lane <  4) vals[t] += vals[t + 4];
        	if (lane <  2) vals[t] += vals[t + 2];
        	if (lane <  1) vals[t] += vals[t + 1];
		__syncthreads();
        
        	// Write result
        	if (lane == 0)
        	{
            		d_out[row] = vals[t];
        	}
    	}	
}

template <typename T>
void spmv_csr_vector(MatrixInfo<T> * mat,T *vector,T *out) 
{
    	T *d_vector,*d_val, *d_out;
    	int *d_cols, *d_ptr;
    	float time_taken;
    	double gflop = 2 * (double) mat->nz / 1e9;
    	float milliseconds = 0;
    	cudaEvent_t start, stop;
    	cudaEventCreate(&start);
    	cudaEventCreate(&stop);

	// Allocate memory on device
    	cudaMalloc(&d_vector,mat->N*sizeof(T));
    	cudaMalloc(&d_val,mat->nz*sizeof(T));
    	cudaMalloc(&d_out,mat->M*sizeof(T));
    	cudaMalloc(&d_cols,mat->nz*sizeof(int));
    	cudaMalloc(&d_ptr,(mat->M+1)*sizeof(int));

	// Copy from host memory to device memory
    	cudaMemcpy(d_vector,vector,mat->N*sizeof(T),cudaMemcpyHostToDevice);
    	cudaMemcpy(d_val,mat->val,mat->nz*sizeof(T),cudaMemcpyHostToDevice);
    	cudaMemcpy(d_cols,mat->cIndex,mat->nz*sizeof(int),cudaMemcpyHostToDevice);
    	cudaMemcpy(d_ptr,mat->rIndex,(mat->M+1)*sizeof(int),cudaMemcpyHostToDevice);
    	cudaMemset(d_out, 0, mat->M*sizeof(T));
  
	// Run the kernel and time it 
    	cudaEventRecord(start); 
    	for (int i = 0; i < ITER; i++)
   		spmv_csr_vector_kernel<T><<<ceil(mat->M / ((float)BlockDim/32)),BlockDim>>>(d_val,d_vector,d_cols,d_ptr,mat->M,d_out);
    	cudaEventRecord(stop);
    	cudaEventSynchronize(stop);
    	cudaEventElapsedTime(&milliseconds, start, stop);

	// Copy from device memory to host memory 
    	cudaMemcpy(out, d_out, mat->M*sizeof(T), cudaMemcpyDeviceToHost);

	// Free device memory
    	cudaFree(d_vector);
    	cudaFree(d_val);
    	cudaFree(d_cols);
    	cudaFree(d_ptr); 
    	cudaFree(d_out);
   
	// Calculate and print out GFLOPs and GB/s 
	double gbs = ((mat->N * sizeof(T)) + (mat->nz*sizeof(T)) + (mat->M*sizeof(int)) + (mat->nz*sizeof(int)) + (mat->M*sizeof(T))) / (milliseconds/ITER) / 1e6;
    	time_taken = (milliseconds/ITER)/1000.0;
    	printf("Average time taken for %s is %f\n", "SpMV by GPU CSR Vector Algorithm",time_taken);
    	printf("Average GFLOP/s is %lf\n",gflop/time_taken);
	printf("Average GB/s is %lf\n\n",gbs);
}
