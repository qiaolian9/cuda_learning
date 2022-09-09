#include<cuda_runtime.h>
#include<stdio.h>
#include<sys/time.h>

#define CHECK(call)                                                                      \
{                                                                                        \
    const cudaError_t error = call;                                                        \
    if (error != cudaSuccess)                                                             \
    {                                                                                    \
        printf("Error: %s:%d \n",__FILE__,__LINE__);                                     \
        printf("code:%d, reason %s\n",error, cudaGetErrorString(error));                 \
        exit(1);                                                                         \
    }                                                                                    \
}                                                                                        \

double cpuSecond(){
    struct timeval tp;
    gettimeofday(&tp,NULL);
    return ((double)tp.tv_sec * 1.0E3 + (double)tp.tv_usec * 1.0E-3);
}

void initialData(float *ip,const int N){
    time_t t;
    srand((unsigned) time(&t));
    for(int i=0;i<N;i++){
        ip[i] = (float)(rand() &0xFF) / 10.0f;
    }
    return ;
}

void checkResults(float *hostRef, float *gpuRef, const int N){
    bool match = 1;
    double epsilon = 1.0E-9;
    for(int i=0;i<N;i++){
        if(abs(hostRef[i] - gpuRef[i]) > epsilon){
            match = 0;
            printf("dont match, error id %d, hostRef %5.2f, gpuRef %5.2f\n",i,hostRef[i],gpuRef[i]);
            break;
        }
    }
    if(match) printf("Arrays match!\n");
    return ;
}

__global__
void sumArrayOnGPU(float *d_A, float *d_B, float *d_C, const int N){
    int index = threadIdx.x;
    int block = blockIdx.x;
    int id = block * blockDim.x + index;
    int x_total = blockDim.x * gridDim.x;
    int cycle_x = (N + x_total - 1) / x_total;
    for(int i=0;i<cycle_x;i++){
        id += i * x_total;
        if(id < N){
            d_C[id] = d_A[id] + d_B[id];
        }
    }
    return ;
}

void sumArray(float *A, float *B, float *hostRef, const int N){
    for(int i=0;i<N;i++){
        hostRef[i] = B[i] + A[i];
    }
}

int main(int argc,char **argv){
    double iStart,iElaps;
    printf("%s starting...\n",argv[0]);
    // initial Device
    cudaDeviceProp deviceProp;
    int dev = 3;
    CHECK(cudaGetDeviceProperties(&deviceProp,dev));
    printf("Using Device %d : %s\n",dev,deviceProp.name);
    CHECK(cudaSetDevice(dev));

    // initial Data
    int n = 24;
    // scanf("%d",&n);
    const int N = 1<<n;
    printf("arrays size is %d\n",N);
    int size = N * sizeof(float);

    // initial host memory
    float *A, *B, *hostRef, *gpuRef;
    A = (float*)malloc(size);
    B = (float*)malloc(size);
    hostRef = (float*)malloc(size);
    gpuRef = (float*)malloc(size);
    initialData(A,N);
    initialData(B,N);
    initialData(hostRef,N);
    initialData(gpuRef,N);

    // initial device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc((float**)&d_A,size);
    cudaMalloc((float**)&d_B,size);
    cudaMalloc((float**)&d_C,size);
    cudaMemcpy(d_A,A,size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,B,size,cudaMemcpyHostToDevice);

    // host code
    iStart = cpuSecond();
    sumArray(A,B,hostRef,N);
    iElaps = cpuSecond() - iStart;
    printf("host code time cost %f ms\n",iElaps);

    // device code
    dim3 block(256);
    dim3 grid((N / 2 + block.x - 1) / block.x);
    iStart = cpuSecond();
    sumArrayOnGPU<<<grid,block>>>(d_A,d_B,d_C,N);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;
    printf("Matrix add cuda time cost %f ms\n",iElaps);
    printf("func<<<(%d %d):(%d %d)>>>\n",grid.x,grid.y,block.x,block.y);

    cudaMemcpy(gpuRef,d_C,size,cudaMemcpyDeviceToHost);
    checkResults(hostRef,gpuRef,N);

    // free memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(A);
    free(B);
    free(hostRef);
    free(gpuRef);
    cudaDeviceReset();
    return 0;
}