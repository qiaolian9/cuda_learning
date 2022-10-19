/* 固定内存：Device端可以直接访问Host端固定内存中的数据、也可通过cudaMemcpy进行数据拷贝到device端的global中
    设计原因：Host端内存采用可分页的内存（虚拟地址），GPU无法得知Host端操作系统执行的状态；
    完成memcpy时：1.开辟固定内存--->2.将可分页内存中数据拷贝到固定内存中--->3.将固定内存中数据拷贝到Device的global memory中
*/
#include<cuda_runtime.h>
#include<stdio.h>
#include<stdlib.h>
#include<iostream>

template<typename T>
__global__
void memoryRead(T *devData, const int n){
    // thread info
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int tid = bx * blockDim.x + tx;
    for(int i = 0 ; i < n ; i++){
        printf("%f ",devData[i]);
    }
    return ;
}

int main(int argc, char** argv){
    printf("%s staring...\n",argv[0]);
    int dev = atoi(argv[1]);
    cudaDeviceProp deviceProp;
    if(cudaGetDeviceProperties(&deviceProp,dev) != cudaSuccess){
        printf("Error in deviceProp\n");
        exit(0);
    }
    cudaSetDevice(dev);
    printf("Using Device %d : %s\n",dev,deviceProp.name);

    const int n = 1 << 3;
    size_t nBytes = n * sizeof(float);
    float *dA, *hA ;
    cudaMalloc((void**)(&dA),nBytes);
    // normal mode
    hA = (float*)malloc(nBytes);
    // pinned memory
    cudaMallocHost((void**)(&hA),nBytes);
    
    // initial data
    for(int i = 0 ; i  < n ; i++)   
        hA[i] = (float)(i);

    cudaMemcpy(dA,hA,nBytes,cudaMemcpyHostToDevice);
    dim3 grid(1);
    dim3 block(1);
    memoryRead<float><<<grid,block>>>(hA,n);
    
    // normal mode
    free(hA);
    // pinned mode
    // cudaFreeHost(hA);

    cudaFree(dA);
    cudaDeviceReset();
    return 0;
}