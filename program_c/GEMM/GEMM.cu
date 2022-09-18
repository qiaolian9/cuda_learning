#include<stdio.h>
#include <stdlib.h>
#include<cuda_runtime.h>
#include "./func/metric.h"

template<typename T>
void M(T **ptr, size_t n){*ptr = (T*)malloc(n);}

// kernel 1 : normal GEMM
template<typename T>
__global__
void SimpleGemm(
    T *__restrict__ A, 
    T *__restrict__ B, 
    T *__restrict__ C,
    const int m, 
    const int n, 
    const int k){
    // block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // global index
    int x = tx + bx * blockDim.x;
    int y = ty + by * blockDim.y;
    int index = y * n + x;

    if(x >= m || y >= n) return;
    // simple loop
    C[index] = 0;
    for(int i=0;i<k;i++){
        C[index] += A[y*n+i] * B[i*n+x];
    }
    return ;
}

template<typename T>
void cpuGemm(
    T *__restrict__ h_A,
    T *__restrict__ h_B,
    T *__restrict__ h_C,
    const int m,
    const int n,
    const int k){
    for(int i=0;i<m;i++){
        for(int j=0;j<n;j++){
            int index = i * n + j;
            h_C[index] = 0;
            for(int l=0;l<k;l++){
                h_C[index] += h_A[i*n+l] * h_B[l*n+j];
            }
        }
    }
    return ;
}

int main(int argc, char **argv){
    if(argc != 4){
        printf("Usage: ./GEMM M N K\n");
        exit(0);
    }
    int dev = 3;
    cudaSetDevice(dev);
    
    // initial Data
    int m, n, k;
    m = 1 <<  atoi(argv[1]);
    n = 1 <<  atoi(argv[2]);
    k = 1 <<  atoi(argv[3]);
    double flopsPerMatrixMul = 2.0 * m * n * k, gigaFlops;
    printf("%s starting with size (M,N,K) : (%s,%s,%s); Data size :  %d %d %d\n",argv[0],argv[1],argv[2],argv[3],m,n,k);
    size_t nBytesA = m * k * sizeof(float);
    size_t nBytesB = k * n * sizeof(float);
    size_t nBytesC = m * n * sizeof(float); 
    float *h_A, *h_B, *h_C, *d_A, *d_B, *d_C, *tmp;

    M<float>(&h_A,nBytesA);
    M<float>(&h_B,nBytesB);
    M<float>(&h_C,nBytesC);
    M<float>(&tmp,nBytesC);

    cudaMalloc((float**)&d_A,nBytesA);
    cudaMalloc((float**)&d_B,nBytesB);
    cudaMalloc((float**)&d_C,nBytesC);

    initialData(h_A,m*k);
    initialData(h_B,k*n);
    cudaMemcpy(d_A,h_A,nBytesA,cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,h_B,nBytesB,cudaMemcpyHostToDevice);


    // initial record tool
    const char* s = "cpu-GEMM";
    int nIter = 1000;
    double iStart, iElaps;
    float gElaps;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // cpu GEMM
    iStart = cpuMsecond();
    for(int i=0;i<nIter;i++){
        cpuGemm<float>(h_A,h_B,tmp,m,n,k);
    }
    iElaps = (cpuMsecond() - iStart) / (float)nIter;
    gigaFlops = (flopsPerMatrixMul*1e-9) / (iElaps * 1e-3);
    printf("%s Time= %f ms, Performance= %f GFlops/s\n",s,iElaps,gigaFlops);

    // simple GEMM
    s = "SimpleGemm";
    dim3 block(32,32);
    dim3 grid((m + block.x - 1) / block.x,(n + block.y - 1) / block.y);
    cudaEventRecord(start);
    for(int i=0;i<nIter;i++){
        SimpleGemm<float><<<grid,block>>>(d_A,d_B,d_C,m,n,k);
        cudaDeviceSynchronize();
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gElaps,start,stop);
    gigaFlops = (flopsPerMatrixMul*1e-9) / (gElaps * 1e-3);
    printf("%s<grid(%d,%d),block(%d,%d)> Time= %f ms, Performance= %f GFlops/s\n",s,grid.x,grid.y,block.x,block.y,gElaps,gigaFlops);
    cudaMemcpy(h_C,d_C,nBytesC,cudaMemcpyDeviceToHost);
    checkResults<float>(tmp,h_C,m*n);
    // free memory
    free(h_A);
    free(h_B);
    free(h_C);
    free(tmp);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    cudaDeviceReset();
    return 0;
}