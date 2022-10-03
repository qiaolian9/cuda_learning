#include<stdio.h>
#include<cuda_runtime.h>

int main(int argc,char **argv){
    if(argc != 5){
        printf("Usage: ./GEMM M N K dev\n");
        exit(0);
    }
    printf("%s starting...\n",argv[0]);
    int dev = 3;
    cudaDeviceProp deviceProp;
    cudaGetDeviceproperties(&deviceProp,dev);
    printf("Using Device %d : %s\n",dev,deviceProp.name);
    cudaSetDevice(dev);

    // initialData
    

    return 0;
}