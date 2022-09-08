#include<iostream>
#include<sys/time.h>
#include<stdlib.h>
__global__
void sumArray(float *A, float *B, float *C, const int N){
    int index = threadIdx.x;
    int block = blockIdx.x;
    int i = block * blockDim.x + index;
    if(i<N) C[i] = A[i] + B[i];
    return ;
}

void initialData(float *ip, const int N){
    time_t t;
    srand((unsigned int) time(&t));

    for(int i=0; i<N; i++){
        ip[i] = (float)(rand() & 0xFF) / 10.0f;
    }
    return ;

}

int main(){
    const int N = 1<<20;
    float *A = (float*)malloc(N * sizeof(float));                  
    float *B = (float*)malloc(N * sizeof(float));
    float *C = (float*)malloc(N * sizeof(float));

    // initial array 
    initialData(A,N);
    initialData(B,N);

    float *d_A, *d_B, *d_C;
    int size = N * sizeof(float);
    cudaMalloc((void**)&d_A,size);
    cudaMalloc((void**)&d_B,size);
    cudaMalloc((void**)&d_C,size);

    // memory copy
    cudaMemcpy(d_A,A,size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,B,size,cudaMemcpyHostToDevice);

    // init CUDA
    dim3 num_threads(256);
    dim3 num_Blocks((N + 255) / 256);
    // func_Ptr
    void (*func)(float*, float*, float*, const int);
    func = sumArray;
    
    func<<<num_Blocks,num_threads>>>(d_A,d_B,d_C,N);

    cudaMemcpy(C,d_C,size,cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(A);
    free(B);
    free(C);

    return 0;
}