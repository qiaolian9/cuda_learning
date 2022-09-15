#include<stdio.h>
#include<cuda_runtime.h>
#include "./func/metric.h"
#include "./func/func.h"
#define M(x,n){x = (int*)malloc(n);}
#define cudaM(x,n){cudaMalloc((void**)&x,n);}

void reduceNeighbored_cpu(int *tmp, unsigned int n, int stride){
    if(stride >= n) return;
    for(int i=0;i<n;i+=stride){
        tmp[i] += tmp[i+stride];
    }
    stride <<= 1;
    reduceNeighbored_cpu(tmp,n,stride);
}

__global__
void nestedRecursiveReduce(int *g_idata, int *g_odata, unsigned int n, int stride){
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
        nestedRecursiveReduce<<<1,stride>>>(blockPtr,o,n,stride);
        cudaDeviceSynchronize();
    }
    __syncthreads();
}

__global__
void nestedRecursiveReduceNosync(int *g_idata, int *g_odata, unsigned int n, int stride){
    unsigned int tid = threadIdx.x;
    // unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int *blockPtr = g_idata + blockIdx.x * blockDim.x;
    int *o = g_odata + blockIdx.x;

    if(stride==1 && tid==0){
        o[tid] = blockPtr[0] + blockPtr[1];
        return;
    }
    if(tid < stride) blockPtr[tid] += blockPtr[tid + stride];
    stride >>= 1;
    if(tid==0){
        nestedRecursiveReduceNosync<<<1,stride>>>(blockPtr,o,n,stride);
    }
}

int main(int argc, char **argv){
    double iStart, iElaps;
    void (*p)(int *, int *, unsigned int, int);
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

    const char *s = "reduceNeighbored_cpu";
    // reduceNeighbered_cpu
    iStart = cpuMSecond();
    reduceNeighbored_cpu(tmp,n,1);
    iElaps = cpuMSecond() - iStart;
    printf("%s time cost %f ms\n",s,iElaps);

    // nestedReduce
    s = "nestedRecursiveReduce";
    p = nestedRecursiveReduce;
    int gpu_sum;
    gpu_sum = func_nested(p,g_idata,g_odata,block.x/2,h_idata,h_odata,n,nBytes,block,grid,s);
    checkResults(gpu_sum,tmp[0]);
    
    // nestedRecursiveReduceNosync
    s = "nestedRecursiveReduceNosync";
    p = nestedRecursiveReduceNosync;
    gpu_sum = func_nested(p,g_idata,g_odata,block.x/2,h_idata,h_odata,n,nBytes,block,grid,s);
    checkResults(gpu_sum,tmp[0]);


    // free memory
    free(h_idata);
    free(h_odata);
    free(tmp);
    cudaFree(g_idata);
    cudaFree(g_odata);

    return 0;
}