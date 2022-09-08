#include<cuda_runtime.h>
#include<stdio.h>

int main(){
    // define total elements
    int nElem = 1024;
    
    // define grid and block structure
    dim3 block (1024);
    dim3 grid ((nElem + block.x - 1) / block.x);
    printf("block nums %d, threads nums %d\n",grid.x,block.x);

    // reset block
    block.x = 512;
    grid.x = (nElem + block.x - 1) / block.x;
    printf("block nums %d, threads nums %d\n",grid.x,block.x);

    // reset block
    block.x = 256;
    grid.x = (nElem + block.x - 1) / block.x;
    printf("block nums %d, threads nums %d\n",grid.x,block.x);
    cudaDeviceSynchronize();
    cudaDeviceReset();
    return 0;
}