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
    printf("设备名称与型号: %s\n", deviceProp.name);
    printf("显存大小: %d MB\n", (int)(deviceProp.totalGlobalMem / 1024 / 1024));
    printf("含有的SM数量: %d\n", deviceProp.multiProcessorCount);
    printf("CUDA CORE数量: %d\n", deviceProp.multiProcessorCount * 64);
    printf("计算能力: %d.%d\n", deviceProp.major, deviceProp.minor);
    printf("gpu clock rate: %.0fMHz\n", deviceProp.clockRate * 1e-3f);
    printf("FP32 %.2fTFlops\n",deviceProp.clockRate * 1e-9f * deviceProp.multiProcessorCount * 64 * 2);
    return 0;
}