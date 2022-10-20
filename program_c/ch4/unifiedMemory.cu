/*  UMA:cuda6.0 unified memory access
    统一内存寻址，基于UVA：托管内存池由系统控制，与zerocopy内存相比（始终驻留在host端），会自动进行数据物理位置的转移
    cudaError_t cudaMallocManaged(void **devPtr, size_t count, unsigned int flags=0)
*/
#include<cuda_runtime.h>
#include<stdio.h>
#include<stdlib.h>
#include<iostream>
template<typename T>
__global__
void uMAInit(T *__restrict__ devPtr, unsigned int n){
    for(int i = 0 ; i < n ; i++){
        devPtr[i] = (T)(i);
    }
}

int main(int argc, char **argv){
    // environment init
    printf("%s starting...\n",argv[0]);
    int dev, n;
    if(argc != 3){
        printf("please use %s [dev] [n]\n",argv[0]);
        exit(0);
    }
    dev = atoi(argv[1]);
    n = 1 << atoi(argv[2]);
    cudaDeviceProp deviceProp;
    if(cudaGetDeviceProperties(&deviceProp, dev) != cudaSuccess){
        std::cout << "Error in device information check!" << std::endl;
        exit(0);
    }
    cudaSetDevice(dev);

    // experience
    size_t nBytes = n * sizeof(float);
    float *dataPtr;
    cudaMallocManaged((void**)&dataPtr,nBytes);

    uMAInit<float><<<1,1>>>(dataPtr, n);
    cudaDeviceSynchronize();

    for(int i = 0 ; i < n ; i++){
        std::cout << dataPtr[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}