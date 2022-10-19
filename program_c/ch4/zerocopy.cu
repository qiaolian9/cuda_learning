#include<cuda_runtime.h>
#include<stdio.h>
#include<stdlib.h>
#include<iostream>


// template<typename T>
// void check(T *hA, T *dA, const int n){
//     for(int i = 0 ; i < n ; i++){
//         if(abs(hA[i] - dA[i]) > 1e-6){
//             std::cout << "error in " << i << std::endl;
//             break;
//         }
//     }
// }

template<typename T>
void allReduceHost(T *hostData, const int n){
    for(int i = 1 ; i < n ; i++){
        hostData[0] += hostData[i];
    }
}


template<typename T>
__global__
void allReduce(T *g_idata, T *g_odata, unsigned int n){
    unsigned int tid = threadIdx.x;
    T *blockPtr = g_idata + blockDim.x * blockIdx.x;
    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if(idx >= n) return;
    for(int stride = 1 ; stride < blockDim.x ; stride *= 2){
        int index = tid * stride * 2;
        if(index < blockDim.x){
            blockPtr[index] += blockPtr[index + stride];
        }
        __syncthreads();
    }
    if(tid == 0) g_odata[blockIdx.x] = blockPtr[tid];
    return ;
}

int main(int argc, char** argv){
    printf("%s starting...\n",argv[0]);
    int dev = atoi(argv[1]);
    cudaDeviceProp deviceProp;
    if(cudaGetDeviceProperties(&deviceProp,dev) != cudaSuccess){
        std::cout << "error in deviceinfo" << std::endl;
        exit(0);
    }
    printf("Using Device %d : %s\n",dev,deviceProp.name);
    cudaSetDevice(dev);
    
    int n = atoi(argv[2]);
    n = 1 << n;
    size_t nBytes = n * sizeof(float);
    dim3 block(512);
    dim3 grid((n + block.x - 1) / block.x);

    // normal mode: host memory ---cudamemcpy---> device global memory
    float *hA, *dB;
    hA = (float*)malloc(nBytes);
    for(int i = 0 ; i < n ; i++){
        hA[i] = 1.0f;
    }
    cudaMalloc((void**)&dB,nBytes);

    // // normal mode
    // float *hB, *dC;
    // hB = (float*)malloc(grid.x*sizeof(float));
    // cudaMalloc((void**)&dC,grid.x * sizeof(float));

    // zerocopy mode
    float *dC;
    cudaHostAlloc((void**)&dC,grid.x*sizeof(float),cudaHostAllocDefault);

    cudaMemcpy(dB,hA,nBytes,cudaMemcpyHostToDevice);

    // calculate
    allReduceHost<float>(hA,n);
    allReduce<float><<<grid,block>>>(dB,dC,n);
    cudaDeviceSynchronize();
    // // normal mode
    // cudaMemcpy(hB,dC,grid.x*sizeof(float),cudaMemcpyDeviceToHost);
    float sumA = 0;
    for(int i = 0 ; i < grid.x ; i++){
        sumA += dC[i];
    }
    std::cout << sumA << " " << hA[0] << ":" << (sumA == hA[0]) << std::endl;

    return 0;
}