#include<cuda_runtime.h>
#include<stdio.h>

#define CHECK(call)                                                                      \
{                                                                                        \
    const cudaError_t error = call;                                                      \
    if (error != cudaSuccess)                                                            \
    {                                                                                    \
        printf("Error: %s:%d \n",__FILE__,__LINE__);                                     \
        printf("code:%d, reason %s\n",error, cudaGetErrorString(error));                 \
        exit(1);                                                                         \
    }                                                                                    \
}                                                                                        \

void checkResults(float *hostRef, float *gpuRef, const int N){
    double epsilon = 1.0E-8;
    bool match = true;
    for(int i=0;i<N;i++){
        if(abs(hostRef[i] - gpuRef[i]) > epsilon){
            match = 0;
            printf("dont match\n");
            printf("%5.2f,%5.2f, index at %d\n",hostRef[i],gpuRef[i],i);
            break;
        }
    }
    if(match) printf("match\n");
}

void initialData(float *ip, int size){
    time_t t;
    srand((unsigned) time(&t));
    for(int i=0;i<size;i++){
        ip[i] = (float)(rand() &0xFF) / 10.0f;
    }
    return ;
}

void sumArray(float *A, float *B, float *C, const int N){
    for(int i=0;i<N;i++){
        C[i] = A[i] + B[i];
    }
    return ;
}

__global__
void sumArrayOnGPU(float *A, float *B, float *C){
    int index = threadIdx.x;
    int block = blockIdx.x;
    int i = block * blockDim.x + index;
    C[i] = A[i] + B[i];
    return ;
}

int main(){
    printf("Starting ...\n");
    int dev = 4;
    cudaSetDevice(dev);
    const int N = 1<<25;
    int bytesize = N * sizeof(float);
    
    // init cpu data
    float *A, *B, *hostRef, *gpuRef;
    A = (float*)malloc(bytesize);
    B = (float*)malloc(bytesize);
    hostRef = (float*)malloc(bytesize);
    gpuRef = (float*)malloc(bytesize);

    initialData(A,N);
    initialData(B,N);
    memset(hostRef,0,bytesize);
    memset(gpuRef,0,bytesize);
    
    // init gpu data
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A,bytesize);
    cudaMalloc((void**)&d_B,bytesize);
    cudaMalloc((void**)&d_C,bytesize);
    
    cudaMemcpy(d_A,A,bytesize,cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,B,bytesize,cudaMemcpyHostToDevice);

    // cpu calculate
    sumArray(A,B,hostRef,N);

    // gpu calculate
    dim3 block(256);
    dim3 grid((N + block.x -1) / block.x);
    sumArrayOnGPU<<<grid,block>>>(d_A,d_B,d_C);
    cudaDeviceSynchronize();
    cudaMemcpy(gpuRef,d_C,bytesize,cudaMemcpyDeviceToHost);

    cudaDeviceReset();
    // check
    checkResults(hostRef,gpuRef,N);

    // free memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(A);
    free(B);
    free(hostRef);
    free(gpuRef);

    return 0;
}