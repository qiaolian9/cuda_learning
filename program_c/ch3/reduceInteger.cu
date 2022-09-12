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
    srand((unsigned) tiem(&t));
    for(unsigned int i=0;i<n;i++){
        ip[i] = (int)(rand() & 0xFF);
    }
}

__global__
void reduceNeighbered(int *g_idata, int *g_odata, unsigned int n){
    unsigned int tid = threadIdx.x;
    int *blockPtr = g_idata + blockDim.x * blockIdx.x;

    if(tid >= n) return;
    for(int stride=1;stride<=blockDim.x;stride*=2){
        if(tid % (stride * 2) == 0){
            blockPtr[tid] += blockPtr[tid+stride];
        }
        __syncthreads()''
    }
    if(tid == 0) g_odata[blockIdx.x] = blockPtr[0];
}

int main(int argc, char **argv){
    printf("%s starting...\n",argv[0]);
    cudaDeviceProp deviceProp;
    int dev = 3;
    cudaGetDeviceProperties(&deviceProp,dev);
    printf("Device info %d : %s",dev,deviceProp.name);
    cudaSetDevice(dev);

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

    cudaMemcpy(g_idata,h_idata,nBytes,cudaMemcpyHostToDevice);
    
    


    return 0;

}