#include<stdio.h>
#include <stdlib.h>
#include<cuda_runtime.h>
#include "./func/metric.h"
#include "cublas_v2.h"
#include<omp.h>

template<typename T>
void M(T **ptr, size_t n){*ptr = (T*)malloc(n);}

// kernel 1 : normal GEMM 
// load M : 2mnk_globalMemory
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

// kernel 2 : block GEMM
// load M : m*n*k*(1/bm + 1/bn)
template<typename T, const int BLOCK_SIZE_M, const int BLOCK_SIZE_N, const int BLOCK_SIZE_K>
__global__
void blockGemm(
    T *__restrict__ A, 
    T *__restrict__ B, 
    T *__restrict__ C,
    const int m, 
    const int n, 
    const int k){
    // shared memory && register
    __shared__ T As[BLOCK_SIZE_M][BLOCK_SIZE_K];
    __shared__ T Bs[BLOCK_SIZE_K][BLOCK_SIZE_N];
    T Cs[BLOCK_SIZE_M][BLOCK_SIZE_N];
    // global index
    int ix = (blockIdx.x * blockDim.x + threadIdx.x) * BLOCK_SIZE_N;
    int iy = (blockIdx.y * blockDim.y + threadIdx.y) * BLOCK_SIZE_M;
    // initial Bc
    for(int jy=0;jy<BLOCK_SIZE_M;jy++){
        for(int jx=0;jx<BLOCK_SIZE_N;jx++){
            Cs[jy][jx] = 0;
        }
    }
    for(int i=0;i<k/BLOCK_SIZE_K;i++){
        // load As
        for(int jy=0;jy<BLOCK_SIZE_M;jy++){
            for(int jx=0;jx<BLOCK_SIZE_K;jx++){
                As[jy][jx] = A[(iy+jy)*k+i * BLOCK_SIZE_K+jx];
            }
        }
        // load Bs
        for(int jy=0;jy<BLOCK_SIZE_K;jy++){
            for(int jx=0;jx<BLOCK_SIZE_N;jx++){
                Bs[jy][jx] = B[(i*BLOCK_SIZE_K+jy)*n + ix+jx];
            }
        }
        // calculate
        for(int jy=0;jy<BLOCK_SIZE_M;jy++){
            for(int jx=0;jx<BLOCK_SIZE_N;jx++){
                for(int jk=0;jk<BLOCK_SIZE_K;jk++){
                    Cs[jy][jx] += As[jy][jk] * Bs[jk][jx];
                }
            }
        }
    }
    int index = iy * n + ix;
    for(int i=0;i<BLOCK_SIZE_M*BLOCK_SIZE_N;i++) C[index++] = Cs[i/BLOCK_SIZE_N][i%BLOCK_SIZE_N];
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
    #pragma omp parallel for
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
    if(argc != 5){
        printf("Usage: ./GEMM M N K dev\n");
        exit(0);
    }
    int dev = atoi(argv[4]);
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
    int nIter = 100;
    double iStart, iElaps;
    float gElaps;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // cpu GEMM
    iStart = cpuMsecond();
    for(int i=0;i<10;i++){
        cpuGemm<float>(h_A,h_B,tmp,m,n,k);
    }
    iElaps = (cpuMsecond() - iStart) / (float)nIter;
    gigaFlops = (flopsPerMatrixMul*1e-9) / (iElaps * 1e-3);
    printf("%s Time= %f ms, Performance= %f GFlops/s\n",s,iElaps / 10.0f,gigaFlops);

    // cuBLAS
    s = "cuBLAS";
    dim3 block(32,32);
    dim3 grid((n + block.x - 1) / block.x,(m + block.y - 1) / block.y);
    cublasHandle_t blas_handle;  
    cublasCreate(&blas_handle);
    float alpha = 1.0;
    float beta = 0;
    cudaEventRecord(start);
    for (int i=0;i<nIter;i++ ) {
        cublasSgemm (blas_handle, CUBLAS_OP_N, CUBLAS_OP_N, 
            m, n, k, &alpha, 
            d_A, k, d_B, n, &beta, d_C, n
        );
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gElaps,start,stop);
    gigaFlops = (flopsPerMatrixMul*1e-9) / (gElaps * 1e-3);
    printf("%s<grid(%d,%d),block(%d,%d)> Time= %f ms, Performance= %f GFlops/s\n",s,grid.x,grid.y,block.x,block.y,gElaps,gigaFlops);
    cudaMemcpy(h_C,d_C,nBytesC,cudaMemcpyDeviceToHost);
    checkResults<float>(tmp,h_C,m*n);

    // simple GEMM
    s = "SimpleGemm";
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

    // blockGemm
    s = "blockGemm";
    const int bm = 4;
    const int bn = 4;
    const int bk = 4;
    grid.x = (n + bn * block.x -1) / (bn * block.x);
    grid.y = (m + bm * block.y -1) / (bm * block.y);
    cudaEventRecord(start);
    for(int i=0;i<nIter;i++){
        blockGemm<float,bm,bn,bk><<<grid,block>>>(d_A,d_B,d_C,m,n,k);
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