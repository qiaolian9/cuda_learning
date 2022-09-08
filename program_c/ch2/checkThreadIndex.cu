#include<cuda_runtime.h>
#include<stdio.h>

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

void initialData(int *ip, const int N){
    for(int i=0;i<N;i++){
        ip[i] = i;
    }
}

void printMatrix(int *A, const int nx, const int ny){
    for(int i=0;i<ny;i++){
        for(int j=0;j<nx;j++){
            printf("%3d",(A+i*nx)[j]);
        }
        printf("\n");
    }
    printf("\n");
}

__global__
void printThreadIndex(int *A, const int nx, const int ny){
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int index = iy * nx + ix;
    printf("threadid: (%d %d); blockid: (%d %d); coordinate: (%d %d); global index %d ival %d\n" \
            ,threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,ix,iy,index,A[index]);
}

int main(int argc, char **argv){
    printf("%s starting...\n",argv[0]);
    int dev = 3;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp,dev));
    printf("Using device %d : %s\n",dev,deviceProp.name);
    CHECK(cudaSetDevice(dev));

    int nx = 8, ny = 6;
    int nxy = nx * ny;
    int nBytes = nxy * sizeof(int);

    int *A = (int*)malloc(nBytes);
    initialData(A,nxy);

    int *d_A;
    cudaMalloc((void**)&d_A,nBytes);
    cudaMemcpy(d_A,A,nBytes,cudaMemcpyHostToDevice);

    printf("matrix\n");
    printMatrix(A,nx,ny);

    // print Thread index
    dim3 block(4,2);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

    printf("ThreadIndex\n");
    printThreadIndex<<<grid,block>>>(d_A,nx,ny);
    cudaDeviceSynchronize();
    
    // free memory
    free(A);
    cudaFree(d_A);

    cudaDeviceReset();
    return 0;
}