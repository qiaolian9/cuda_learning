#include<stdio.h>
#include<cuda_runtime.h>

__device__ float devData;

__global__
void checkGlobalvariable(){
    printf("Device: the value of the global variable is %f\n",devData);
    devData += 2.0f;
}

int main(){
    float value = 3.14f;
    // M1 cudaMemcpyToSymbol
    cudaMemcpyToSymbol(devData,&value,sizeof(float));
    // M2 cudaMemcpy
    // float *dPtr = NULL;
    // cudaGetSymbolAddress((void**)&dPtr,devData);
    // cudaMemcpy(dPtr,&value,sizeof(float),cudaMemcpyHostToDevice);
    printf("Host: copied %f to the global memory\n",value);

    checkGlobalvariable<<<1,1>>>();
    cudaDeviceSynchronize();
    // M1
    cudaMemcpyFromSymbol(&value,devData,sizeof(float));
    // M2  
    // cudaMemcpy(&value,dPtr,sizeof(float),cudaMemcpyDeviceToHost);
    printf("Host: the value changed by the kernel to %f\n",value);
    cudaDeviceReset();
    return 0;
}