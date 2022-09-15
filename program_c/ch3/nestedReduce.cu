#include<stdio.h>
#include<cuda_runtime.h>
#include "./func/metric.h"
#define M(x,n){x = (int*)malloc(n);}
#define cudaM(x,n){cudaMalloc((void**)&x,n);}

void reduceNeighbered_cpu(int *tmp, unsigned int n, int stride){
    if(stride >= n) return;
    for(int i=0;i<n;i+=stride){
        tmp[i] += tmp[i+stride];
    }
    stride <<= 1;
    reduceNeighbered_cpu(tmp,n,stride);
}

__global__
void nestedReduce(int *g_idata, int *g_odata, unsigned int n, int stride){
    unsigned int tid = threadIdx.x;
    // unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int *blockPtr = g_idata + blockIdx.x * blockDim.x;
    int *o = g_odata + blockIdx.x;

    if(stride==1 && tid==0){
        o[tid] = blockPtr[0] + blockPtr[1];
        return;
    }
    if(tid < stride) blockPtr[tid] += blockPtr[tid + stride];
    __syncthreads();
    stride >>= 1;
    if(tid==0){
        nestedReduce<<<1,stride>>>(blockPtr,o,n,stride);
        cudaDeviceSynchronize();
    }
    __syncthreads();
}

int main(int argc, char **argv){
    double iStart, iElaps;
    printf("%s starting...\n",argv[0]);
    int dev = 3;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp,dev);
    printf("Using Device %d : %s\n",dev,deviceProp.name);

    // initial Data
    int n = 1 << 25;
    dim3 block(512);
    dim3 grid((n + block.x - 1) / block.x);
    size_t nBytes = n * sizeof(int);
    size_t oBytes = grid.x * sizeof(int);
    int *h_idata, *h_odata, *tmp;
    M(h_idata,nBytes);
    M(h_odata,oBytes);
    M(tmp,nBytes);
    initialData(h_idata,n);
    memcpy(tmp,h_idata,nBytes);
    int *g_idata, *g_odata;
    cudaM(g_idata,nBytes);
    cudaM(g_odata,oBytes);
    cudaMemcpy(g_idata,h_idata,nBytes,cudaMemcpyHostToDevice);

    const char *s = "reduceNeighbered_cpu";
    // reduceNeighbered_cpu
    iStart = cpuMSecond();
    reduceNeighbered_cpu(tmp,n,1);
    iElaps = cpuMSecond() - iStart;
    printf("%s time cost %f ms\n",s,iElaps);

    // nestedReduce
    s = "nestedReduce";
    int gpu_sum = 0;
    iStart = cpuMSecond();
    nestedReduce<<<grid,block>>>(g_idata,g_odata,n,block.x/2);
    cudaMemcpy(h_odata,g_odata,oBytes,cudaMemcpyDeviceToHost);
    for(int i=0;i<grid.x;i++){
        gpu_sum += h_odata[i];
    }
    iElaps = cpuMSecond() - iStart;
    printf("%s cuda time cost %f ms;",s,iElaps);
    printf("func <<<%d , %d>>>\n",grid.x,block.x);
    checkResults(gpu_sum,tmp[0]);
    
    // free memory
    free(h_idata);
    free(h_odata);
    free(tmp);
    cudaFree(g_idata);
    cudaFree(g_odata);

    return 0;
}