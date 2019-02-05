#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <cuda.h>
#include "mmio.h"

template <typename T>
void spmv_cpu(T *val, T *vec, int *cols, int *rowDelimiters, int N, T *out)
{
    	for (int i = 0; i < N; i++)
    	{
        	T t = 0;
        	for (int j = rowDelimiters[i]; j < rowDelimiters[i + 1]; j++)
        	{
            		int col = cols[j];
            		t += val[j] * vec[col];
        	}
        	out[i] = t;
    	}
}

