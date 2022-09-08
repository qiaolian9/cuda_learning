#include<cuda_runtime.h>
#include<stdio.h>
__global__
void checkDimension(){
    printf("thread (%d %d %d), block (%d %d %d), blockDim (%d %d %d), gridDim (%d %d %d)\n",
                threadIdx.x,threadIdx.y,threadIdx.z,
                blockIdx.x,blockIdx.y,blockIdx.z,
                blockDim.x,blockDim.y,blockDim.z,
                gridDim.x,gridDim.y,gridDim.z);
    return ;
}

int main(){
    int nElem = 6;
    // define grid and block
    dim3 block (3);
    dim3 grid ((nElem + block.x - 1) / block.x);

    // check grid and block dimension from host
    printf("grid.x %d, grid.y %d, grid.z %d\n",grid.x,grid.y,grid.z);
    printf("block.x %d, block.y %d, block.z %d\n",block.x,block.y,block.z);

    // check grid and block dimension from device
    checkDimension<<<grid,block>>>();

    // reset device before you leave
    cudaDeviceReset();
    return 0;

}