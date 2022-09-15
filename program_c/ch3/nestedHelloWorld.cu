#include<stdio.h>
#include<cuda_runtime.h>

__global__
void nestedHelloWorld(int iSize, int iDepth){
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    printf("Recursion=%d, HW from thread %d block %d\n",iDepth,tid,bid);
    // __syncthreads();
    iSize >>= 1;
    if(tid == 0 && iSize > 0 && bid == 0){
        printf("-------->nested execution depth : %d\n",iDepth);
        nestedHelloWorld<<<2,iSize>>>(iSize,++iDepth);
        cudaDeviceSynchronize();
    }
}
int main(int argc, char **argv){
    printf("%s staring...\n",argv[0]);
    int dev=3;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp,dev);
    cudaSetDevice(dev);
    printf("Using Device %d : %s\n",dev,deviceProp.name);

    dim3 block(8);
    nestedHelloWorld<<<2,block>>>(block.x,0);
    cudaDeviceSynchronize();
    return 0;
}