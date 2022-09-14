#include<cuda_runtime.h>
#include<stdio.h>
#include "metric.h"

int func(void (*p)(int *g_idata, int *g_odata, unsigned int n), int *g_idata, int *g_odata,
        int *h_idata, int *h_odata, unsigned int n, size_t nBytes, dim3 block, dim3 grid, const char* s){
    double iStart, iElaps;
    cudaMemcpy(g_idata,h_idata,nBytes,cudaMemcpyHostToDevice);
    iStart = cpuMSecond();
    (*p)<<<grid,block>>>(g_idata,g_odata,n);
    cudaDeviceSynchronize();
    cudaMemcpy(h_odata,g_odata,grid.x * sizeof(int),cudaMemcpyDeviceToHost);
    int gpu_sum = 0;
    for(int i=0;i<grid.x;i++){
        gpu_sum += h_odata[i];
    }
    iElaps = cpuMSecond() - iStart;
    printf("%s cuda time cost %f ms & ",s,iElaps);
    printf("func<<<(%d %d):(%d %d)>>>\n",grid.x,grid.y,block.x,block.y);
    return gpu_sum;
}