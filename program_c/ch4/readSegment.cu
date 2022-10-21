/*  readSegment.cu: 非对齐内存访问对访存io性能的影响
    global memory ---> L1 cache(32B) ---> L2 cache(128B)
    CUDA执行调度单元为warp(32threads)，每个线程产生一个访存需求！！！（对齐内存访问+合并内存访问）
    1.对齐内存访问：设备内存事务的第一个访存地址为缓存粒度的整数倍；
    2.合并内存访问：warp产生的访存内一块连续的内存区域
    对齐+合并+4B/thread = 总线访存事务利用率100%
    eg: C[i] = A[i] + B[i]
*/
#include<cuda_runtime.h>
#include<stdio.h>
#include<stdlib.h>
#include<iostream>
#include "./func/func.h"

#define M(ptr,n) {ptr = (float*)malloc(n);}
#define cM(ptr,n) {cudaMalloc((void**)&ptr,n);}


template<typename T>
void sumArraysOnHost(T *__restrict__ hC, T *__restrict__ hA, T *__restrict__ hB, unsigned int n, int offset){
    for(int i = offset ; i < n ; i++){
        hC[i] = hA[i] + hB[i];
    }
    return ;
}

template<typename T>
__global__
void sumArrays(T *__restrict__ devC, T *__restrict__ devA, T *__restrict__ devB, unsigned int n, int offset){
    // thread information
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int tid = bx * blockDim.x + tx;
    int idx = tid * 4 + offset;

    if(idx >= n) return;
    for(int i = 0 ; i < 4 ; i++)
        devC[idx + i] = devA[idx + i] + devB[idx + i];
}

int main(int argc, char **argv){
    // set up environment
    printf("%s starting...\n",argv[0]);
    if(argc != 4){
        printf("please use ./readSegment [dev] [n] [offset]\n");
        exit(0);
    }
    int dev = atoi(argv[1]);
    int n = 1 << atoi(argv[2]);
    int offset = atoi(argv[3]);
    int nIter = 5000;
    cudaDeviceProp deviceProp;
    if(cudaGetDeviceProperties(&deviceProp, dev) != cudaSuccess){
        printf("Error in device info checking...\n");
        exit(0);
    }
    cudaSetDevice(dev);
    float gElaps;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // init data
    unsigned int nBytes = n * sizeof(float);
    float *A, *B, *C, *hostRef, *gpuA, *gpuB, *gpuC;
    // memory alloc
    M(A,nBytes);
    M(B,nBytes);
    M(C,nBytes);
    M(hostRef,nBytes);
    cM(gpuA,nBytes);
    cM(gpuB,nBytes);
    cM(gpuC,nBytes);
    initData(A, n);
    memcpy(B, A, nBytes);
    cudaMemcpy(gpuA, A, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(gpuB, B, nBytes, cudaMemcpyHostToDevice);

    // exp excuate
    dim3 block(512);
    dim3 grid((n - offset + block.x - 1) / (block.x * 4));
    sumArraysOnHost<float>(C, A, B, n, offset);
    cudaEventRecord(start);
    for(int i = 0 ; i < nIter ; i++){
        sumArrays<float><<<grid,block>>>(gpuC, gpuA, gpuB, n, offset);
        cudaDeviceSynchronize();
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gElaps,start,stop);
    gElaps /= nIter;
    printf("<grid(%d,%d),block(%d,%d)> Time= %f ms\n", grid.x,grid.y,block.x,block.y,gElaps);

    // result check
    cudaMemcpy(hostRef, gpuC, nBytes, cudaMemcpyDeviceToHost);
    resCheck(C, hostRef, n);

    // free memory
    free(A);
    free(B);
    free(C);
    free(hostRef);
    cudaFree(gpuA);
    cudaFree(gpuB);
    cudaFree(gpuC);
    cudaDeviceReset();
    return 0;
}