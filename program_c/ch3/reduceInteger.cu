#include<cuda_runtime.h>
#include<stdio.h>
#include<sys/time.h>

double cpuMSecond(){
    struct timeval t;
    gettimeofday(&t,NULL);
    return ((double)t.tv_sec * 1.0E3 + (double)t.tv_usec * 1.0E-3);
}

void initialData(int *ip,const unsigned int n){
    time_t t;
    srand((unsigned) time(&t));
    for(unsigned int i=0;i<n;i++){
        ip[i] = (int)(rand() & 0xFF);
    }
}

void checkResults(int x, int y){
    if(x == y){
        printf("results match!!!\n");
        return ;
    }
    printf("results error!!!\n");
    return ;
}

void reduceNeighbered_cpu(int *tmp, unsigned int n,unsigned int stride){
    if(stride>=n){
        return;
    }
    for(int i=0;i<n;i+=stride){
        tmp[i] += tmp[i+stride];
    }
    stride *= 2;
    reduceNeighbered_cpu(tmp,n,stride);
}

__global__
void reduceNeighbered(int *g_idata, int *g_odata, unsigned int n){
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
void reduceNeighbered_2(int *g_idata, int *g_odata, unsigned int n){
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


int main(int argc, char **argv){
    double iStart, iElaps;
    printf("%s starting...\n",argv[0]);
    cudaDeviceProp deviceProp;
    int dev = 3;
    cudaGetDeviceProperties(&deviceProp,dev);
    printf("Device info %d : %s\n",dev,deviceProp.name);
    cudaSetDevice(dev);

    int n = 1 << 25;
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
    
    // reduceInteger cpu
    iStart = cpuMSecond();
    reduceNeighbered_cpu(tmp,n,1);
    iElaps = cpuMSecond() - iStart;
    printf("reduceInteger time cost : %f ms\n",iElaps);

    // reduceInteger gpu
    cudaMemcpy(g_idata,h_idata,nBytes,cudaMemcpyHostToDevice);
    iStart = cpuMSecond();
    reduceNeighbered<<<grid,block>>>(g_idata,g_odata,n);
    cudaDeviceSynchronize();
    cudaMemcpy(h_odata,g_odata,grid.x * sizeof(int),cudaMemcpyDeviceToHost);
    int gpu_sum = 0;
    for(int i=0;i<grid.x;i++) gpu_sum += h_odata[i];
    iElaps = cpuMSecond() - iStart;
    printf("reduceInteger cuda time cost %f ms\n",iElaps);
    printf("func<<<(%d %d):(%d %d)>>>\n",grid.x,grid.y,block.x,block.y);
    checkResults(gpu_sum,tmp[0]);

    // reduceInteger_2 gpu
    cudaMemcpy(g_idata,h_idata,nBytes,cudaMemcpyHostToDevice);
    iStart = cpuMSecond();
    reduceNeighbered_2<<<grid,block>>>(g_idata,g_odata,n);
    cudaDeviceSynchronize();
    cudaMemcpy(h_odata,g_odata,grid.x * sizeof(int),cudaMemcpyDeviceToHost);
    gpu_sum = 0;
    for(int i=0;i<grid.x;i++) gpu_sum += h_odata[i];
    iElaps = cpuMSecond() - iStart;
    printf("reduceInteger_2 cuda time cost %f ms\n",iElaps);
    printf("func<<<(%d %d):(%d %d)>>>\n",grid.x,grid.y,block.x,block.y);
    checkResults(gpu_sum,tmp[0]);

    // free host/device memory
    free(h_idata);
    free(h_odata);
    free(tmp);
    cudaFree(g_idata);
    cudaFree(g_odata);
    cudaDeviceReset();

    return 0;

}