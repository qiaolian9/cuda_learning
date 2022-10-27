#include<stdio.h>

#define T 512
#define B 10
#define C 16

__global__
void cal(int **W, int ***K, int ***x){
    int block = blockIdx.x;
    int index = threadIdx.x;
    int b = block / C, c = block % C;
    // printf("block is %d; thread is %d K is %d\n",block,index,K[b][c][index]);
    x[b][c][index] = 0;
    for(int l=0;l<index;l++){
        x[b][c][index] += W[c][l] * K[b][c][l];
    }
    return ;
}

int main(){
    // int W[C][T], K[B][C][T], x[B][C][T];
    int **W, ***K, ***x;
    cudaMallocManaged(&W, C * sizeof(int*));
    // init W
    for(int i = 0; i < C; i++){
        // allocate unified memoty
        cudaMallocManaged((W+i),T * sizeof(int));
        for(int j = 0; j < T; j++){
            W[i][j] = i * T + j;
        }
    }
    cudaMallocManaged(&K,B * sizeof(int**));
    cudaMallocManaged(&x,B * sizeof(int**));
    // init K
    for(int i = 0; i < B; i++){
        cudaMallocManaged((K+i),C * sizeof(int*));
        cudaMallocManaged((x+i),C * sizeof(int*));
        for(int j = 0; j < C; j++){
            // allocate unified memoty
            cudaMallocManaged((*(x+i)+j), T * sizeof(int));
            cudaMallocManaged((*(K+i)+j), T * sizeof(int));
            for(int k = 0; k < T; k++){
                K[i][j][k] = i * (T * C) + j * T + k;
            }
        }
    }
    // calculate

    dim3 num_Blocks(B*C);
    dim3 num_threads(T);
    
    cal<<<num_Blocks,num_threads>>>(W,K,x);
    cudaDeviceSynchronize();
    
    // free memory
    for(int i=0;i<B;i++){
        for(int j=0;j<C;j++){
            cudaFree(x[i][j]);
            cudaFree(K[i][j]);
        }
    }
    for(int i=0;i<B;i++){
        cudaFree(x[i]);
        cudaFree(K[i]);
    }
    for(int i=0;i<C;i++) cudaFree(W[i]);
    cudaFree(x);
    cudaFree(K);
    cudaFree(W);
    
    return 0;
}