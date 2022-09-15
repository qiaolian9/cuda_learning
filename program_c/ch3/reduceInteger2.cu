#include<cuda_runtime.h>
#include<stdio.h>
#include "./func/metric.h"
#include "./func/func.h"

void reduceNeighbored_cpu(int *tmp, unsigned int n, unsigned int stride){
    if(stride>=n){
        return;
    }
    for(int i=0;i<n;i+=stride){
        tmp[i] += tmp[i+stride];
    }
    stride *= 2;
    reduceNeighbored_cpu(tmp,n,stride);
}

__global__
void reduceNeighbored(int *g_idata, int *g_odata, unsigned int n){
    unsigned int tid = threadIdx.x;
    int *blockPtr = g_idata + blockDim.x * blockIdx.x;
    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if(idx >= n) return;
    for(int stride=1;stride<blockDim.x;stride*=2){
        if(tid % (stride * 2) == 0){
            blockPtr[tid] += blockPtr[tid+stride];
        }
        __syncthreads();
    }
    if(tid == 0) g_odata[blockIdx.x] = blockPtr[tid];
}

__global__
void reduceNeighbored_2(int *g_idata, int *g_odata, unsigned int n){
    unsigned int tid = threadIdx.x;
    int *blockPtr = g_idata + blockDim.x * blockIdx.x;
    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if(idx >= n) return;
    for(int stride=1;stride<blockDim.x;stride*=2){
        int index = tid * 2 * stride;
        if(index < blockDim.x){
            blockPtr[index] += blockPtr[index+stride];
        }
        __syncthreads();
    }
    if(tid == 0) g_odata[blockIdx.x] = blockPtr[tid];
}

__global__
void reduceInterleaved(int *g_idata, int *g_odata, unsigned int n){
    unsigned int tid = threadIdx.x;
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int *blockPtr = g_idata + blockIdx.x * blockDim.x;
    if(idx >= n) return ;
    for(int stride=blockDim.x/2;stride>=1;stride/=2){
        if(tid < stride){
            blockPtr[tid] += blockPtr[tid+stride];
        }
        __syncthreads();
    }
    if(tid == 0) g_odata[blockIdx.x] = blockPtr[0];
    return ;
}

__global__
void reduceUnrolling8(int *g_idata, int *g_odata, unsigned int n){
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockDim.x * blockIdx.x * 8 + threadIdx.x;
    int *blockPtr = g_idata + blockDim.x * blockIdx.x * 8;
    if(idx >= n) return ;
    // unrolling 8 data block
    for(int stride=1;stride<8;stride++){
        if(tid+stride*blockDim.x < n)
            blockPtr[tid] += blockPtr[tid+stride*blockDim.x];
    }
    __syncthreads();
    // loop interleaved
    for(int stride = blockDim.x/2;stride>32;stride>>=1){
        if(tid < stride){
            blockPtr[tid] += blockPtr[tid+stride];
        }
        __syncthreads();
    }
    // unrolling 32(warp)
    if(tid<32){
        volatile int *vmem = blockPtr;
        // int *vmem = blockPtr;
        vmem[tid] += vmem[tid+32];
        vmem[tid] += vmem[tid+16];
        vmem[tid] += vmem[tid+8];
        vmem[tid] += vmem[tid+4];
        vmem[tid] += vmem[tid+2];
        vmem[tid] += vmem[tid+1];
    }
    if(tid==0) g_odata[blockIdx.x] = blockPtr[0];
}

__global__
void reduceCompleteUnrollWarp8(int *g_idata, int *g_odata, unsigned int n){
    unsigned int tid = threadIdx.x;
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x * 8;
    int *blockPtr = g_idata + blockIdx.x * blockDim.x * 8;
    if(idx >= n) return;
    // unroll 8 (1 block -> 8 data block)
    // unrolling 8 data block
    for(int stride=1;stride<8;stride++){
        if(tid+stride*blockDim.x < n)
            blockPtr[tid] += blockPtr[tid+stride*blockDim.x];
    }
    __syncthreads();
    // completeUnroll
    if(blockDim.x >= 1024 && tid < 512)
        blockPtr[tid] += blockPtr[tid+512];
    __syncthreads();
    if(blockDim.x >= 512 && tid < 256)
        blockPtr[tid] += blockPtr[tid+256];
    __syncthreads();
    if(blockDim.x >= 256 && tid <128)
        blockPtr[tid] += blockPtr[tid+128];
    __syncthreads();
    if(blockDim.x >= 128 && tid <64)
        blockPtr[tid] += blockPtr[tid+64];
    __syncthreads();
    // unroll warp
    if(tid<32){
        volatile int* vmem = blockPtr;
        vmem[tid] += vmem[tid+32];
        vmem[tid] += vmem[tid+16];
        vmem[tid] += vmem[tid+8];
        vmem[tid] += vmem[tid+4];
        vmem[tid] += vmem[tid+2];
        vmem[tid] += vmem[tid+1];
    }
    if(tid==0) g_odata[blockIdx.x] = blockPtr[tid];
    return ;
}

int main(int argc, char **argv){
    double iStart, iElaps;
    printf("%s starting...\n",argv[0]);
    cudaDeviceProp deviceProp;
    int dev = 3;
    cudaGetDeviceProperties(&deviceProp,dev);
    printf("Device info %d : %s\n",dev,deviceProp.name);
    cudaSetDevice(dev);
    // kernel funcPtr
    void (*p)(int *, int *, unsigned int);

    int n = 1 << 24;
    // init cuda kernel config
    dim3 block(512);
    dim3 grid((n + block.x - 1) / block.x);

    // init data & malloc host/device memory
    size_t nBytes = n * sizeof(int);
    int *h_idata = (int*)malloc(nBytes);
    int *h_odata = (int*)malloc(grid.x * sizeof(int));
    int *tmp = (int*)malloc(nBytes);
    initialData(h_idata,n);
    memcpy(tmp,h_idata,nBytes);

    int *g_idata, *g_odata;
    cudaMalloc((void**)&g_idata,nBytes);
    cudaMalloc((void**)&g_odata,grid.x * sizeof(int));
    
    // reduceNeighbored_cpu
    iStart = cpuMSecond();
    reduceNeighbored_cpu(tmp,n,1);
    iElaps = cpuMSecond() - iStart;
    printf("reduceNeighbored_cpu time cost : %f ms\n",iElaps);

    int gpu_sum;
    // reduceNeighbored gpu
    p = reduceNeighbored;
    const char* s = "reduceNeighbored";
    gpu_sum = func(p,g_idata,g_odata,h_idata,h_odata,n,nBytes,block,grid,s);
    checkResults(gpu_sum,tmp[0]);
    printf("%d %d\n",gpu_sum,tmp[0]);

    // reduceNeighbored_2 gpu
    p = reduceNeighbored_2;
    s = "reduceNeighbored_2";
    gpu_sum = func(p,g_idata,g_odata,h_idata,h_odata,n,nBytes,block,grid,s);
    checkResults(gpu_sum,tmp[0]);
    printf("%d %d\n",gpu_sum,tmp[0]);
    
    // reduceInterleaved gpu
    p = reduceInterleaved;
    s = "reduceInterleaved";
    gpu_sum = func(p,g_idata,g_odata,h_idata,h_odata,n,nBytes,block,grid,s);
    checkResults(gpu_sum,tmp[0]);
    printf("%d %d\n",gpu_sum,tmp[0]);

    // reduceUnrolling8 gpu
    p = reduceUnrolling8;
    s = "reduceUnrolling8";
    grid.x /= 8;
    gpu_sum = func(p,g_idata,g_odata,h_idata,h_odata,n,nBytes,block,grid,s);
    checkResults(gpu_sum,tmp[0]);
    printf("%d %d\n",gpu_sum,tmp[0]);

    // reduceCompleteUnrollWarp8
    p = reduceCompleteUnrollWarp8;
    s = "reduceCompleteUnrollWarp8";
    gpu_sum = func(p,g_idata,g_odata,h_idata,h_odata,n,nBytes,block,grid,s);
    checkResults(gpu_sum,tmp[0]);
    printf("%d %d\n",gpu_sum,tmp[0]);


    // free host/device memory
    free(h_idata);
    free(h_odata);
    free(tmp);
    cudaFree(g_idata);
    cudaFree(g_odata);
    cudaDeviceReset();

    return 0;

}