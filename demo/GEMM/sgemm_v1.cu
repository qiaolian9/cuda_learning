#include<cuda_runtime.h>
#include<stdio.h>
#include<stdlib.h>

int main(int argc,char** argv){
    if(argc != 5){
        printf("please use ./gemm [M] [N] [K] [device_id]\n");
        exit(0);
    }
    printf("%s Starting...\n",argv[0]);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp,atoi(argv[4]));
    return 0;
}