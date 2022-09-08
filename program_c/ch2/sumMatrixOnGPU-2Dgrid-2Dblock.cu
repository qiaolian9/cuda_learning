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
}   

void initialData(float *ip, const unsigned int nxy){
    time_t t;
    srand((unsigned) time(&t));
    for(unsigned int i=0;i<nxy;i++){
        ip[i] = (float)(rand() & 0xFF) / 10.0f;
    }
}                                                                                     

void checkResult(float *hostRef, float *gpuRef, const int nx, const int ny){
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
    }
    if(match) printf("Matrix Match!\n");
}

double cpuSecond(){
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
    unsigned int index = iy * nx + ix;
    if(ix < nx && iy < ny){
        d_C[index] = d_A[index] + d_B[index];
    }
}

int main(int argc, char **argv){
    // initial environment
    printf("%s starting...\n",argv[0]);
    int dev = 3;
    cudaDeviceProp deviceProp;
    CHECK(cudaDeviceProperties(&deviceProp,dev));
    printf("Using Device %d : %s\n",dev,deviceProp.name);
    CHECK(cudaSetDevice(dev));

    // initial data
    int nx = 1 << 15;
    int ny = 1 << 15;
    unsigned int nxy = nx * ny;

    // initial host memory
    float *A, *B, *hostRef, *gpuRef;
    A = (float*)malloc(nxy * sizeof(float));
    B = (float*)malloc(nxy * sizeof(float));
    hostRef = (float*)malloc(nxy * sizeof(float));
    gpuRef = (float*)malloc(nxy * sizeof(float));
    initialData(A,nxy);
    initialData(B,nxy);
    initialData(hostRef,nxy);
    initialData(gpuRef);

    // initial device memory
    float *d_A, *d_B, *d_C;


    float 

    return 0;
}