<h1>Introduction</h1>

Sparse matrix vector multiplication (SpMV) is of significant importance to computing in a number of scientific and engineering disciplines. However due to the arbitrary sparsitypatterns and sizes of sparse matrices, the parellisation of SpMV is still beset with many operational issues including poor memory coalescing, thread divergence and load imbalance. I implement five different GPU based algorithms in CUDA and analyze their performance on different types and sizes of data. I try to draw out insights on the strengths and weaknesses of these algorithms and the situations in which they are best used. I look at performance in terms of computational throughput, memory bandwidth utilization and other system generated metrics.

<h1>Implementation</h1>

To compile, simply run 
make

To run the program, type
./spmv <input data matrix file name>

To download data, refer to Matrices.pdf

<h1>Requirements </h1>

cuda-9.1
compute capability >= 5.2
