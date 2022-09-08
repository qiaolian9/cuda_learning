#include<cuda_runtime.h>
#include<stdio.h>
#include<sys/time.h>


#define CHECK(call)                                                                      \
{                                                                                        \
    const cudaError error = call;                                                        \
    if (error != cudaSuccess)                                                            \
    {                                                                                    \
        printf("Error: %s:%d \n",__FILE__,__LINE__);                                     \
        printf("code:%d, reason %s\n",error, cudaGetErrorString(error));                 \
        exit(1);                                                                         \
    }                                                                                    \
}                                                                                        \

void initialData(float *ip, const int nxy){
    time_t t;
    srand((unsigned) time(&t));
    for(int i=0;i<nxy;i++){
        ip[i] = (float)(rand() & 0xFF) / 10.0f;
    }
}                                                                                     

void checkResults(float *hostRef, float *gpuRef, const int nx, const int ny){
    bool match = 1;
    double epsilon = 1.0E-9;
    for(int i=0;i<ny;i++){
        for(int j=0;j<nx;j++){
            int row = i * nx;
            if(abs((hostRef + row)[j] - (gpuRef + row)[j]) > epsilon){
                match = 0;
                printf("dont match!!! Error index (%d %d), host %5.2f, device %5.2f\n"  \
                        ,i,j,(hostRef + row)[j],(gpuRef + row)[j]);
                break;
            }
        }
        if(!match) break;
    }
    if(match) printf("Matrix Match!\n");
}

double cpuMSecond(){
    struct timeval tp;
    gettimeofday(&tp,NULL);
    return ((double)tp.tv_sec * 1.0E3 + (double)tp.tv_usec * 1.0E-3);
}

void sumMatrix(float *A, float *B, float *hostRef, const int nx, const int ny){
    for(int i=0;i<ny;i++){
        for(int j=0;j<nx;j++){
            (hostRef + i * nx)[j] = (B + i * nx)[j] + (A + i * nx)[j];
        }
    }
}

__global__
void sumMatrixOnGPU(float *d_A, float *d_B, float *d_C, const int nx, const int ny){
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    int index = iy * nx + ix;
    if(ix < nx && iy < ny){
        d_C[index] = d_A[index] + d_B[index];
    }
    return ;
}

int main(int argc, char **argv){
    // initial environment
    double iStart, iElaps;
    printf("%s starting...\n",argv[0]);
    int dev = 3;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp,dev));
    printf("Using Device %d : %s\n",dev,deviceProp.name);
    CHECK(cudaSetDevice(dev));

    // initial data
    int n;
    scanf("%d",&n);
    int nx = 1 << n;
    int ny = 1 << n;
    int nxy = nx * ny;
    int nBytes = nxy * sizeof(float);

    // initial host memory
    float *A, *B, *hostRef, *gpuRef;
    A = (float*)malloc(nBytes);
    B = (float*)malloc(nBytes);
    hostRef = (float*)malloc(nBytes);
    gpuRef = (float*)malloc(nBytes);
    initialData(A,nxy);
    initialData(B,nxy);

    // initial device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, nBytes);
    cudaMalloc((void**)&d_B, nBytes);
    cudaMalloc((void**)&d_C, nBytes);

    cudaMemcpy(d_A,A,nBytes,cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,B,nBytes,cudaMemcpyHostToDevice);

    // host code run
    iStart = cpuMSecond();
    sumMatrix(A,B,hostRef,nx,ny);
    iElaps = cpuMSecond() - iStart;
    printf("Matrix add time cost time %f ms\n",iElaps);

    // device code run
    dim3 block(32,32);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);
    
    iStart = cpuMSecond();
    sumMatrixOnGPU<<<grid,block>>>(d_A,d_B,d_C,nx,ny);
    cudaDeviceSynchronize();
    iElaps = cpuMSecond() - iStart;
    printf("Matrix add cuda(2Dgrid-2Dblock) time cost %f ms\n",iElaps);

    cudaMemcpy(gpuRef,d_C,nBytes,cudaMemcpyDeviceToHost);
    checkResults(hostRef,gpuRef,nx,ny);

    // free memory
    free(A);
    free(B);
    free(hostRef);
    free(gpuRef);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaDeviceReset();

    return 0;
}