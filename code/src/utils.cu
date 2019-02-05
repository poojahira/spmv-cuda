#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>
#include <float.h>
#include <math.h>
#include "mmio.h"

/* Code for next five functions sourced from https://github.com/nathanMiniovich/cuda-spmv. Used with some modifications. */

template <typename T>
int cmpTuple(const void * a, const void * b){
	return (*(((MTuple<T> *)a)->r) - *(((MTuple<T> *)b)->r));
}

template <typename T>
MatrixInfo<T> * read_file(const char * path){
	MM_typecode matcode;
	FILE *f;
	int M, N, nz, i;   //M is row number, N is column number and nz is the number of entry
	int *rIndex, *cIndex;
	T *val;

	if ((f = fopen(path, "r")) == NULL)
	{
        return NULL;
	}
	if (mm_read_banner(f, &matcode) != 0)
	{
        return NULL;
	}

	/*  This is how one can screen matrix types if their application */
	/*  only supports a subset of the Matrix Market data types.      */
	if (mm_is_complex(matcode) && mm_is_matrix(matcode) &&
		mm_is_sparse(matcode))
	{
        return NULL;
	}

	/* find out size of sparse matrix .... */
	if ((mm_read_mtx_crd_size(f, &M, &N, &nz)) != 0)
        return NULL;

	/* reseve memory for matrices */
	rIndex = (int *)malloc(nz * sizeof(int));
	cIndex = (int *)malloc(nz * sizeof(int));
	val = (T *)malloc(nz * sizeof(T));

	/* NOTE: when reading in floats, ANSI C requires the use of the "l"  */
	/*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
	/*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */
	for (i = 0; i<nz; i++)
	{
        double tmp;
		fscanf(f, "%d %d %lg\n", &rIndex[i], &cIndex[i], &tmp);
		rIndex[i]--;  /* adjust from 1-based to 0-based */
		cIndex[i]--;
        val[i] = (T) tmp;
	}

	if (f != stdin) fclose(f);

    MatrixInfo<T> * mat_inf = (MatrixInfo<T> *) malloc(sizeof(MatrixInfo<T>));
    mat_inf->M = M;
    mat_inf->N = N;
    mat_inf->nz = nz;
    mat_inf->rIndex = rIndex;
    mat_inf->cIndex = cIndex;
    mat_inf->val = val;
    return mat_inf;
}


template <typename T>
void freeMatrixInfo(MatrixInfo<T> * inf){
    free(inf->rIndex);
    free(inf->cIndex);
    free(inf->val);
    free(inf);
}

template <typename T>
MatrixInfo<T> * transferMat(MatrixInfo<T> * mat){

	MTuple<T> * tupleMat = (MTuple<T> *) malloc(mat->nz*sizeof(MTuple<T>));

	for (int i = 0 ; i < mat->nz ; i++){
		tupleMat[i].r = &(mat->rIndex[i]);
		tupleMat[i].c = &(mat->cIndex[i]);
		tupleMat[i].v = &(mat->val[i]);
	}
	qsort(tupleMat, mat->nz, sizeof(MTuple<T>), cmpTuple<T>);
	
        MatrixInfo<T> * newMat = (MatrixInfo<T> *) malloc(sizeof(MatrixInfo<T>));
	newMat->rIndex = (int *) malloc(mat->nz*sizeof(int));
	newMat->cIndex = (int *) malloc(mat->nz*sizeof(int));
	newMat->val = (T *) malloc(mat->nz*sizeof(T));

	newMat->M = mat->M;
	newMat->N = mat->N;
	newMat->nz = mat->nz;
	
	for(int i = 0 ; i < mat->nz ; i++){
		newMat->rIndex[i] = *(tupleMat[i].r);
		newMat->cIndex[i] = *(tupleMat[i].c);
		newMat->val[i] = *(tupleMat[i].v);
	}
	
	free(tupleMat);
	
	return newMat;
}

template <typename T>
void convert2CSR(MatrixInfo<T> * mat){
	int * CSRIndex = (int *) malloc((mat->M+1)*sizeof(int));
	// takes values 0 to M
	int currRow = 0;
	// number of nonzero elements seen so far
	int count = 0;

	CSRIndex[0] = 0;

	int i = 0;
	while(i < mat->nz){
		int elemRow = mat->rIndex[i];
		
		if( currRow != elemRow ){
				CSRIndex[currRow+1] = count;
				currRow++;
		}else{
			count++;
			i++;
		}
						
	}

	if(count == mat->nz){
		CSRIndex[currRow+1] = count;
	}

	free(mat->rIndex);
	mat->rIndex = CSRIndex;
}


template <typename T>
int verify(int nz, int M, int *rIndex, int *cIndex, T *val, T *vec, T *res) {

	T *correct = (T*)malloc(sizeof(T) * M);
	memset(correct, 0, sizeof(T) * M);
	for (int i = 0; i < nz; ++i) {
		correct[rIndex[i]] += val[i] * vec[cIndex[i]];
	}

	int o = 0;
	for (int i = 0; i < M; i++) {
		if (round(correct[i]) == 0 && round(res[i]) == 0){
			if (fabs(correct[i] - res[i]) > FLT_EPSILON) {
				o++;
                                printf("Yours - %lf, correct - %lf, Absolute error - %lf\n", res[i], correct[i], fabs(correct[i] - res[i]));
                                printf("Row index is: %d\n", i);
			}
		}
		else {
    			if (fabs(correct[i] - res[i])/correct[i] > 0.01)
        		{
				o++;
				printf("Yours - %lf, correct - %lf, Relative error - %lf\n", res[i], correct[i], fabs(correct[i] - res[i])/correct[i]);
                        	printf("Row index is: %d\n", i);
			}
		}

	}
	return o;
}

template <typename T>
T* write_vector(int N) {
	T *vec = (T *) malloc(N*sizeof(T));
	for (int i = 0; i < N; i++){
                vec[i] = 1.0;
        }
	return vec;
}
