CC = nvcc
objects = mmio.o main.o

spmv: $(objects)
        $(CC) -arch=sm_52 $(objects) -o spmv

mmio.o: src/mmio.c
        $(CC) -x cu -arch=sm_52 -Iinclude -c src/mmio.c

main.o: src/spmv_csr_scalar.cu src/spmv_csr_vector.cu src/spmv_csr_adaptive.cu src/spmv_pcsr.cu src/spmv_light.cu src/spmv_cpu.cu src/utils.cu src/main.cu
        $(CC) -arch=sm_52 -Iinclude -I. -dc src/main.cu

