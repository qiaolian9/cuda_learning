#include<stdio.h>
#include <stdlib.h>
#include<cuda_runtime.h>
#include "./func/metric.h"
#include "cublas_v2.h"
#include<omp.h>

template<typename T>
void M(T **ptr, size_t n){*ptr = (T*)malloc(n);}

#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])


// kernel 1 : normal GEMM 
// load M : 2mnk_globalMemory
template<typename T>
__global__
void SimpleGemm(
    T *__restrict__ A, 
    T *__restrict__ B, 
    T *__restrict__ C,
    const int m, 
    const int n, 
    const int k){
    // block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // global index
    int x = tx + bx * blockDim.x;
    int y = ty + by * blockDim.y;
    int index = y * n + x;

    if(x >= m || y >= n) return;
    // simple loop
    C[index] = 0;
    for(int i=0;i<k;i++){
        C[index] += A[y*n+i] * B[i*n+x];
    }
    return ;
}

// kernel 2 : block GEMM
// load M : m*n*k*(1/bm + 1/bn)
template<typename T, const int BLOCK_SIZE_M, const int BLOCK_SIZE_N, const int BLOCK_SIZE_K>
__global__
void blockGemm(
    T *__restrict__ A, 
    T *__restrict__ B, 
    T *__restrict__ C,
    const int m, 
    const int n, 
    const int k){
    // shared memory && register
    __shared__ T As[BLOCK_SIZE_M][BLOCK_SIZE_K];
    __shared__ T Bs[BLOCK_SIZE_K][BLOCK_SIZE_N];
    T Cs[BLOCK_SIZE_M][BLOCK_SIZE_N];
    // global index
    int ix = (blockIdx.x * blockDim.x + threadIdx.x) * BLOCK_SIZE_N;
    int iy = (blockIdx.y * blockDim.y + threadIdx.y) * BLOCK_SIZE_M;
    // initial Bc
    for(int jy=0;jy<BLOCK_SIZE_M;jy++){
        for(int jx=0;jx<BLOCK_SIZE_N;jx++){
            Cs[jy][jx] = 0;
        }
    }
    for(int i=0;i<k/BLOCK_SIZE_K;i++){
        // load As
        for(int jy=0;jy<BLOCK_SIZE_M;jy++){
            for(int jx=0;jx<BLOCK_SIZE_K;jx++){
                As[jy][jx] = A[(iy+jy)*k+i * BLOCK_SIZE_K+jx];
            }
        }
        // load Bs
        for(int jy=0;jy<BLOCK_SIZE_K;jy++){
            for(int jx=0;jx<BLOCK_SIZE_N;jx++){
                Bs[jy][jx] = B[(i*BLOCK_SIZE_K+jy)*n + ix+jx];
            }
        }
        // calculate
        for(int jy=0;jy<BLOCK_SIZE_M;jy++){
            for(int jx=0;jx<BLOCK_SIZE_N;jx++){
                for(int jk=0;jk<BLOCK_SIZE_K;jk++){
                    Cs[jy][jx] += As[jy][jk] * Bs[jk][jx];
                }
            }
        }
    }
    int index = iy * n + ix;
    for(int i=0;i<BLOCK_SIZE_M*BLOCK_SIZE_N;i++) C[index++] = Cs[i/BLOCK_SIZE_N][i%BLOCK_SIZE_N];
    return ;
}

// kernel 3 : block sgemm
template<
    typename T, 
    const int BLOCK_SIZE_M,
    const int BLOCK_SIZE_N,
    const int BLOCK_SIZE_K, 
    const int THREAD_SIZE_Y,
    const int THREAD_SIZE_X
    >
__global__
void sgemm_v1(
    T *__restrict__ A,
    T *__restrict__ B,
    T *__restrict__ C,
    const int M,
    const int N,
    const int K
    ){
    // block index
    int bx = blockIdx.x;
    int by = blockIdx.y;
    // thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    // the number in block of X,Y
    const int THREAD_X_PER_BLOCK = BLOCK_SIZE_N / THREAD_SIZE_X;
    const int THREAD_Y_PER_BLOCK = BLOCK_SIZE_M / THREAD_SIZE_Y;
    const int THREAD_NUM_PER_BLOCK = THREAD_X_PER_BLOCK * THREAD_Y_PER_BLOCK;

    int tid = ty * THREAD_X_PER_BLOCK + tx;
    // shared memory
    __shared__ T As[2][BLOCK_SIZE_K][BLOCK_SIZE_M];
    __shared__ T Bs[2][BLOCK_SIZE_K][BLOCK_SIZE_N];
    // register C memory
    T accum[THREAD_SIZE_Y][THREAD_SIZE_X] = {0};
    // register for A & B
    T frag_A[2][THREAD_SIZE_Y];
    T frag_B[2][THREAD_SIZE_X];
    // register for load A,B
    const int ldg_num_a = BLOCK_SIZE_K * BLOCK_SIZE_M / (THREAD_NUM_PER_BLOCK * 4);
    // const int ldg_num_b = BLOCK_SIZE_K * BLOCK_SIZE_N / (THREAD_NUM_PER_BLOCK * 4);
    T ldg_a_reg[ldg_num_a * 4];
    // T ldg_b_reg[ldg_num_b * 4];

    // threads number in one row
    const int A_TILE_THREAD_PER_ROW = BLOCK_SIZE_K / 4;
    const int B_TILE_THREAD_PER_ROW = BLOCK_SIZE_N / 4;

    // row & col number thread needs to load
    const int A_TILE_ROW_START = tid / A_TILE_THREAD_PER_ROW;
    const int B_TILE_ROW_START = tid / B_TILE_THREAD_PER_ROW;

    const int A_TILE_COL = tid % A_TILE_THREAD_PER_ROW * 4;
    const int B_TILE_COL = tid % B_TILE_THREAD_PER_ROW * 4;

    const int A_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / A_TILE_THREAD_PER_ROW;
    const int B_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / B_TILE_THREAD_PER_ROW;

    A = (A + (by*BLOCK_SIZE_M)*K);
    B = (B + bx * BLOCK_SIZE_N);

    // from global memory load A to shared memory
    #pragma unroll
    for(int i=0;i<BLOCK_SIZE_M;i+=A_TILE_ROW_STRIDE){
        int ldg_index = i / A_TILE_ROW_STRIDE * 4;
        FETCH_FLOAT4(ldg_a_reg[ldg_index]) = FETCH_FLOAT4(A[OFFSET(A_TILE_ROW_START+i,
            A_TILE_COL,
            K)]);
        As[0][A_TILE_COL][A_TILE_ROW_START+i] = ldg_a_reg[ldg_index];
        As[0][A_TILE_COL+1][A_TILE_ROW_START+i] = ldg_a_reg[ldg_index+1];
        As[0][A_TILE_COL+2][A_TILE_ROW_START+i] = ldg_a_reg[ldg_index+2];
        As[0][A_TILE_COL+3][A_TILE_ROW_START+i] = ldg_a_reg[ldg_index+3];
    }
    // from global memory load B to shared memory
    #pragma unroll
    for(int i=0;i<BLOCK_SIZE_K;i+=B_TILE_ROW_STRIDE){
        FETCH_FLOAT4(Bs[0][B_TILE_ROW_START+i][B_TILE_COL]) = FETCH_FLOAT4(B[OFFSET(B_TILE_ROW_START+i,
            B_TILE_COL,
            N)]);
    }
    __syncthreads();

    // load A from shared memory to register
    #pragma unroll
    for(int i=0;i<THREAD_SIZE_Y;i+=4){
        FETCH_FLOAT4(frag_A[0][i]) = FETCH_FLOAT4(As[0][0][ty*THREAD_SIZE_Y+i]);
    }
    // load B from shared memory to register
    #pragma unroll
    for(int i=0;i<THREAD_SIZE_X;i+=4){
        FETCH_FLOAT4(frag_B[0][i]) = FETCH_FLOAT4(Bs[0][0][tx*THREAD_SIZE_X+i]);
    }
    int write_stage_idx = 1;
    int tile_idx = 0;
    do{
        tile_idx += BLOCK_SIZE_K;
        int load_stage_index = write_stage_idx ^ 1;
        // calculate C
        #pragma unroll
        for(int i=1;i<BLOCK_SIZE_K;++i){
            #pragma unroll
            for(int thread_y = 0; thread_y < THREAD_SIZE_Y; thread_y += 4){
                FETCH_FLOAT4(frag_A[i%2][thread_y]) = FETCH_FLOAT4(As[load_stage_index][i][thread_y]);
            }
            #pragma unroll
            for(int thread_x = 0; thread_x < THREAD_SIZE_X; thread_x += 4){
                FETCH_FLOAT4(frag_B[i%2][thread_x]) = FETCH_FLOAT4(Bs[load_stage_index][i][thread_x]);
            }
            #pragma unroll
            for(int thread_y = 0; thread_y < THREAD_SIZE_Y; thread_y++){
                #pragma unroll
                for(int thread_x = 0; thread_x < THREAD_SIZE_X; thread_x++){
                    accum[thread_y][thread_x] += frag_A[(i+1)%2][thread_y] * frag_B[(i+1)%2][thread_x];
                }
            }
        }
        #pragma unroll
        for(int thread_y = 0; thread_y < THREAD_SIZE_Y; thread_y++){
            #pragma unroll
            for(int thread_x = 0; thread_x < THREAD_SIZE_X; thread_x++){
                accum[thread_y][thread_x] += frag_A[1][thread_y] * frag_B[1][thread_x];
            }
        }

        // load next iter date from global to shared memory
        if(tile_idx < K){
            #pragma unroll
            for(int i = 0; i < BLOCK_SIZE_M; i += A_TILE_ROW_STRIDE){
                int ldg_index = i / A_TILE_ROW_STRIDE * 4;
                FETCH_FLOAT4(ldg_a_reg[ldg_index]) = FETCH_FLOAT4(A[OFFSET(A_TILE_ROW_START+i,
                    A_TILE_COL + tile_idx,
                    K)]);
                As[write_stage_idx][A_TILE_COL][A_TILE_ROW_START+i] = ldg_a_reg[ldg_index];
                As[write_stage_idx][A_TILE_COL+1][A_TILE_ROW_START+i] = ldg_a_reg[ldg_index+1];
                As[write_stage_idx][A_TILE_COL+2][A_TILE_ROW_START+i] = ldg_a_reg[ldg_index+2];
                As[write_stage_idx][A_TILE_COL+3][A_TILE_ROW_START+i] = ldg_a_reg[ldg_index+3];
            }
            #pragma unroll
            for(int i = 0; i < BLOCK_SIZE_K; i += B_TILE_ROW_STRIDE){
                FETCH_FLOAT4(Bs[write_stage_idx][B_TILE_ROW_START+i][B_TILE_COL]) = FETCH_FLOAT4(B[OFFSET(
                    B_TILE_ROW_START+i,
                    B_TILE_COL + tile_idx,
                    N)]);
            }
            __syncthreads();
            write_stage_idx ^= 1;
            #pragma unroll
            for(int thread_y = 0; thread_y < THREAD_SIZE_Y; thread_y += 4){
                FETCH_FLOAT4(frag_A[0][thread_y]) = FETCH_FLOAT4(As[load_stage_index^1][0][thread_y]);
            }
            #pragma unroll
            for(int thread_x = 0; thread_x < THREAD_SIZE_X; thread_x += 4){
                FETCH_FLOAT4(frag_B[0][thread_x]) = FETCH_FLOAT4(Bs[load_stage_index^1][0][thread_x]);
            }
        }
    }while(tile_idx < K);

    // store C from register to global
    #pragma unroll
    for(int thread_y = 0; thread_y < THREAD_SIZE_Y; thread_y++){
        #pragma unroll
        for(int thread_x = 0; thread_x < THREAD_SIZE_X; thread_x += 4){
            FETCH_FLOAT4(C[OFFSET(BLOCK_SIZE_M * by + THREAD_SIZE_Y * ty + thread_y,
            BLOCK_SIZE_N * bx + THREAD_SIZE_X * tx + thread_x,
            N)]) = FETCH_FLOAT4(accum[thread_y][thread_x]);
        }
    }
    return ;
}

template<typename T>
void cpuGemm(
    T *__restrict__ h_A,
    T *__restrict__ h_B,
    T *__restrict__ h_C,
    const int m,
    const int n,
    const int k){
    #pragma omp parallel for
    for(int i=0;i<m;i++){
        for(int j=0;j<n;j++){
            int index = i * n + j;
            h_C[index] = 0;
            for(int l=0;l<k;l++){
                h_C[index] += h_A[i*n+l] * h_B[l*n+j];
            }
        }
    }
    return ;
}

int main(int argc, char **argv){
    if(argc != 5){
        printf("Usage: ./GEMM M N K dev\n");
        exit(0);
    }
    int dev = atoi(argv[4]);
    cudaSetDevice(dev);
    
    // initial Data
    int m, n, k;
    m = 1 <<  atoi(argv[1]);
    n = 1 <<  atoi(argv[2]);
    k = 1 <<  atoi(argv[3]);
    double flopsPerMatrixMul = 2.0 * m * n * k, gigaFlops;
    printf("%s starting with size (M,N,K) : (%s,%s,%s); Data size :  %d %d %d\n",argv[0],argv[1],argv[2],argv[3],m,n,k);
    size_t nBytesA = m * k * sizeof(float);
    size_t nBytesB = k * n * sizeof(float);
    size_t nBytesC = m * n * sizeof(float); 
    float *h_A, *h_B, *h_C, *d_A, *d_B, *d_C, *tmp;

    M<float>(&h_A,nBytesA);
    M<float>(&h_B,nBytesB);
    M<float>(&h_C,nBytesC);
    M<float>(&tmp,nBytesC);

    cudaMalloc((float**)&d_A,nBytesA);
    cudaMalloc((float**)&d_B,nBytesB);
    cudaMalloc((float**)&d_C,nBytesC);

    initialData(h_A,m*k);
    initialData(h_B,k*n);
    cudaMemcpy(d_A,h_A,nBytesA,cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,h_B,nBytesB,cudaMemcpyHostToDevice);


    // initial record tool
    const char* s = "cpu-GEMM";
    int nIter = 100;
    // double iStart, iElaps;
    float gElaps;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // cpu GEMM
    // iStart = cpuMsecond();
    // for(int i=0;i<10;i++){
    //     cpuGemm<float>(h_A,h_B,tmp,m,n,k);
    // }
    // iElaps = (cpuMsecond() - iStart) / 10.0f;
    // gigaFlops = (flopsPerMatrixMul*1e-12) / (iElaps * 1e-3);
    // printf("%s Time= %f ms, Performance= %f TFlops/s\n",s,iElaps,gigaFlops);

    // cuBLAS
    s = "cuBLAS";
    dim3 block(32,32);
    dim3 grid((n + block.x - 1) / block.x,(m + block.y - 1) / block.y);
    cublasHandle_t blas_handle;  
    cublasCreate(&blas_handle);
    float alpha = 1.0;
    float beta = 0;
    cudaEventRecord(start);
    for (int i=0;i<nIter;i++ ) {
        cublasSgemm (blas_handle, CUBLAS_OP_N, CUBLAS_OP_N, 
            m, n, k, &alpha, 
            d_A, k, d_B, n, &beta, d_C, n
        );
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gElaps,start,stop);
    cublasDestroy(blas_handle); 
    gElaps /= nIter;
    gigaFlops = (flopsPerMatrixMul*1e-12) / (gElaps * 1e-3);
    printf("%s<grid(%d,%d),block(%d,%d)> Time= %f ms, Performance= %f TFlops/s\n",s,grid.x,grid.y,block.x,block.y,gElaps,gigaFlops);
    cudaMemcpy(tmp,d_C,nBytesC,cudaMemcpyDeviceToHost);
    // checkResults<float>(tmp,h_C,m*n);

    // simple GEMM
    s = "SimpleGemm";
    cudaEventRecord(start);
    for(int i=0;i<nIter;i++){
        SimpleGemm<float><<<grid,block>>>(d_A,d_B,d_C,m,n,k);
        cudaDeviceSynchronize();
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gElaps,start,stop);
    gElaps /= nIter;
    gigaFlops = (flopsPerMatrixMul*1e-12) / (gElaps * 1e-3);
    printf("%s<grid(%d,%d),block(%d,%d)> Time= %f ms, Performance= %f TFlops/s\n",s,grid.x,grid.y,block.x,block.y,gElaps,gigaFlops);
    cudaMemcpy(h_C,d_C,nBytesC,cudaMemcpyDeviceToHost);
    checkResults<float>(tmp,h_C,m*n);

    // blockGemm
    s = "blockGemm";
    grid.x = (n + 4 * block.x -1) / (4 * block.x);
    grid.y = (m + 4 * block.y -1) / (4 * block.y);
    cudaEventRecord(start);
    for(int i=0;i<nIter;i++){
        blockGemm<float,4,4,4><<<grid,block>>>(d_A,d_B,d_C,m,n,k);
        cudaDeviceSynchronize();
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gElaps,start,stop);
    gElaps /= nIter;
    gigaFlops = (flopsPerMatrixMul*1e-12) / (gElaps * 1e-3);
    printf("%s<grid(%d,%d),block(%d,%d)> Time= %f ms, Performance= %f TFlops/s\n",s,grid.x,grid.y,block.x,block.y,gElaps,gigaFlops);
    cudaMemcpy(h_C,d_C,nBytesC,cudaMemcpyDeviceToHost);
    checkResults<float>(tmp,h_C,m*n);
    
    // sgemm_v1
    s = "sgemm_v1";
    const int bm = 128;
    const int bn = 128;
    const int bk = 8;
    const int tm = 8;
    const int tn = 8;
    block.x = bn / tn;
    block.y = bm / tm;
    grid.x = n / bn;
    grid.y = m / bm;
    cudaEventRecord(start);
    for(int i=0;i<nIter;i++){
        sgemm_v1<float,bm,bn,bk,tm,tn><<<grid,block>>>(d_A,d_B,d_C,m,n,k);
        cudaDeviceSynchronize();
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gElaps,start,stop);
    gElaps /= nIter;
    gigaFlops = (flopsPerMatrixMul*1e-12) / (gElaps * 1e-3);
    printf("%s<grid(%d,%d),block(%d,%d)> Time= %f ms, Performance= %f TFlops/s\n",s,grid.x,grid.y,block.x,block.y,gElaps,gigaFlops);
    cudaMemcpy(h_C,d_C,nBytesC,cudaMemcpyDeviceToHost);
    checkResults<float>(tmp,h_C,m*n);

    // free memory
    free(h_A);
    free(h_B);
    free(h_C);
    free(tmp);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    cudaDeviceReset();
    return 0;
}