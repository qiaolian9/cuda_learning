#include<cuda_runtime.h>
#include<stdio.h>

int main(int argc, char **argv){
    // device count
    printf("%s starting...\n",argv[0]);
    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

    if(error_id != cudaSuccess){
        printf("cudaGetDeviceCount returned %d\n -> %s\n",
                (int)error_id,cudaGetErrorString(error_id));
        exit(EXIT_FAILURE);
    }

    if(deviceCount == 0){
        printf("There are no available devices that support CUDA\n");
    }else{
        printf("Detected %d CUDA Capable devices\n",deviceCount);
    }

    // device Info (eg device name) 
    int dev = 3;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp,dev);
    printf("Device  %d : %s\n",dev,deviceProp.name);

    return 0;
}