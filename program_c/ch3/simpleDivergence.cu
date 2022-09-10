#include<stdio.h>
#include<cuda_runtime.h>
#include<sys/time.h>
#include<iostream>

double cpuMSecond(){
    struct timeval tp;
    gettimeofday(&tp,NULL);
    return ((double)tp.tv_sec * 1.0E3 + (double)tp.tv_usec * 1.0E-3);
}

__global__
void mathKernel1(float *C){
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    float a, b;
    a = b = 0.0f;
    if(index % 2 == 0){
        a = 100.0f;
    }else{
        b = 200.0f;
    }
    C[index] = a + b;
    return ;
}

__global__
void mathKernel2(float *C){
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    float a, b;
    a = b = 0.0f;
    if((index / warpSize) % 2 == 0){
        a = 100.0f;
    }else{
        b = 200.0f;
    }
    C[index] = a + b;
    return ;
}

// not concered
// __global__
// void mathKernel3(float *C){
//     int index = threadIdx.x + blockIdx.x * blockDim.x;
//     float a, b;
//     a = b = 0.0f;
//     return ;
// }

int main(int argc, char **argv){
    double iStart, iElaps;
    printf("%s starting...\n",argv[0]);
    int dev = 3;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp,dev);
    printf("Using Device %d : %s\n",dev,deviceProp.name);
    cudaSetDevice(dev);

    // inti data
    int n;
    scanf("%d",&n);
    n = 1<<n;
    int nBytes = n * sizeof(float);
    float *d_C;
    cudaMalloc((void**)&d_C,nBytes);

    dim3 block(64);
    dim3 grid((n + block.x - 1) / block.x);

    // warmup
    mathKernel1<<<grid,block>>>(d_C);
    cudaDeviceSynchronize();
    // mathKernel1
    iStart = cpuMSecond();
    mathKernel1<<<grid,block>>>(d_C);
    cudaDeviceSynchronize();
    iElaps = cpuMSecond() - iStart;
    printf("mathKernel1 cuda time cost %f ms\n",iElaps);
    printf("mathKernel1<<<(%d %d):(%d %d)>>>\n",grid.x,grid.y,block.x,block.y);

    // mathKernel2
    iStart = cpuMSecond();
    mathKernel2<<<grid,block>>>(d_C);
    cudaDeviceSynchronize();
    iElaps = cpuMSecond() - iStart;
    printf("mathKernel2 cuda time cost %f ms\n",iElaps);
    printf("mathKerne2<<<(%d %d):(%d %d)>>>\n",grid.x,grid.y,block.x,block.y);

    // free memory
    cudaFree(d_C);
    cudaDeviceReset();    
    // cudaError_t err = cudaGetLastError();  // add
    // if (err != cudaSuccess) std::cout << "CUDA error: " << cudaGetErrorString(err) << std::endl; // add
    return 0;
}