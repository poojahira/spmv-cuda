#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <cuda.h>
#include <math.h>
#include "mmio.h"

#define threadsPerBlock 64
#define sizeSharedMemory 1024
#define BlockDim 1024
#define ITER 3

template <typename T>
__global__ void spmv_pcsr_kernel1(T * d_val,T * d_vector,int * d_cols,int d_nnz, T * d_v)
{
    	int tid = blockIdx.x * blockDim.x + threadIdx.x;
    	int icr = blockDim.x * gridDim.x;
    	while (tid < d_nnz){
		d_v[tid] = d_val[tid] * d_vector[d_cols[tid]];
        	tid += icr;
    	}
}

template <typename T>
__global__ void spmv_pcsr_kernel2(T * d_v,int * d_ptr,int N,T * d_out)
{
    	int gid = blockIdx.x * blockDim.x + threadIdx.x;
    	int tid = threadIdx.x;
    
    	__shared__ volatile int ptr_s[threadsPerBlock + 1];
    	__shared__ volatile T v_s[sizeSharedMemory];
 
   	// Load ptr into the shared memory ptr_s
    	ptr_s[tid] = d_ptr[gid];

	// Assign thread 0 of every block to store the pointer for the last row handled by the block into the last shared memory location
    	if (tid == 0) { 
    		if (gid + threadsPerBlock > N) {
	    		ptr_s[threadsPerBlock] = d_ptr[N];}
		else {
    	    		ptr_s[threadsPerBlock] = d_ptr[gid + threadsPerBlock];}
    	}
    	__syncthreads();

    	int temp = (ptr_s[threadsPerBlock] - ptr_s[0])/threadsPerBlock + 1;
    	int nlen = min(temp * threadsPerBlock,sizeSharedMemory);
    	T sum = 0;
    	int maxlen = ptr_s[threadsPerBlock];     
    	for (int i = ptr_s[0]; i < maxlen; i += nlen){
    		int index = i + tid;
    		__syncthreads();
    		// Load d_v into the shared memory v_s
    		for (int j = 0; j < nlen/threadsPerBlock;j++){
	    		if (index < maxlen) {
	        		v_s[tid + j * threadsPerBlock] = d_v[index];
	        		index += threadsPerBlock;
            		}
    		}
   	 	__syncthreads();

    		// Sum up the elements for a row
		if (!(ptr_s[tid+1] <= i || ptr_s[tid] > i + nlen - 1)) {
	   		int row_s = max(ptr_s[tid] - i, 0);
	    		int row_e = min(ptr_s[tid+1] -i, nlen);
	    		for (int j = row_s;j < row_e;j++){
				sum += v_s[j];
	    		}
		}	
    	}	
	// Write result
    	d_out[gid] = sum;
}

template <typename T>
void spmv_pcsr(MatrixInfo<T> * mat,T *vector,T *out) 
{
    	T *d_vector,*d_val, *d_out,*d_v;
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
    	cudaMalloc(&d_v,mat->nz*sizeof(T));
    	cudaMalloc(&d_out,mat->M*sizeof(T));
    	cudaMalloc(&d_cols,mat->nz*sizeof(int));
    	cudaMalloc(&d_ptr,(mat->M+1)*sizeof(int));

	// Copy from host memory to device memory
    	cudaMemcpy(d_vector,vector,mat->N*sizeof(T),cudaMemcpyHostToDevice);
    	cudaMemcpy(d_val,mat->val,mat->nz*sizeof(T),cudaMemcpyHostToDevice);
    	cudaMemcpy(d_cols,mat->cIndex,mat->nz*sizeof(int),cudaMemcpyHostToDevice);
    	cudaMemcpy(d_ptr,mat->rIndex,(mat->M+1)*sizeof(int),cudaMemcpyHostToDevice);
    	cudaMemset(d_out, 0, mat->M*sizeof(T));

	// Run the kernels and time them
    	cudaEventRecord(start);
	for (int i = 0; i < ITER; i++) {
    		spmv_pcsr_kernel1<T><<<ceil(mat->nz/(float)BlockDim),BlockDim>>>(d_val,d_vector,d_cols,mat->nz,d_v);
    		spmv_pcsr_kernel2<T><<<ceil(mat->M/(float)threadsPerBlock),threadsPerBlock>>>(d_v,d_ptr,mat->M,d_out);
    	}
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
    	cudaFree(d_v);
  	
	// Calculate and print out GFLOPs and GB/s
	double gbs = ((mat->N * sizeof(T)) + (mat->nz*sizeof(T) * 3) + (mat->M*sizeof(int)) + (mat->nz*sizeof(int)) + (mat->M*sizeof(T))) / (milliseconds/ITER) / 1e6; 
    	time_taken = (milliseconds/ITER)/1000.0; 
    	printf("Average time taken for %s is %f\n", "SpMV by GPU PCSR Algorithm",time_taken);
    	printf("Average GFLOP/s is %lf\n",gflop/time_taken);
	printf("Average GB/s is %lf\n\n",gbs);
}
