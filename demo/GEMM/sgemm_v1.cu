#pragma once

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

// kernel 3 : block sgemm on register
// shared memory load: 2mnk -> mnk(1/rm + 1/rn)
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
    T ldg_a_reg[ldg_num_a * 4];

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

    A = &A[(BLOCK_SIZE_M * by)* K];
    B = &B[BLOCK_SIZE_N * bx];

    // from global memory load A to shared memory
    #pragma unroll
    for(int i=0;i<BLOCK_SIZE_M;i+=A_TILE_ROW_STRIDE){
        int ldg_index = i / A_TILE_ROW_STRIDE * 4;
        FETCH_FLOAT4(ldg_a_reg[ldg_index]) = FETCH_FLOAT4(A[OFFSET(
            A_TILE_ROW_START + i,
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
        FETCH_FLOAT4(Bs[0][B_TILE_ROW_START+i][B_TILE_COL]) = FETCH_FLOAT4(B[OFFSET(
            B_TILE_ROW_START+i,
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
        int load_stage_idx = write_stage_idx ^ 1;
        // calculate C
        #pragma unroll
        for(int i = 0; i < BLOCK_SIZE_K - 1; ++i){
            #pragma unroll
            for(int thread_y = 0; thread_y < THREAD_SIZE_Y; thread_y += 4){
                FETCH_FLOAT4(frag_A[(i+1)%2][thread_y]) = FETCH_FLOAT4(As[load_stage_idx][i+1][thread_y + ty * THREAD_SIZE_Y]);
            }
            #pragma unroll
            for(int thread_x = 0; thread_x < THREAD_SIZE_X; thread_x += 4){
                FETCH_FLOAT4(frag_B[(i+1)%2][thread_x]) = FETCH_FLOAT4(Bs[load_stage_idx][i+1][thread_x + tx * THREAD_SIZE_X]);
            }
            #pragma unroll
            for(int thread_y = 0; thread_y < THREAD_SIZE_Y; thread_y++){
                #pragma unroll
                for(int thread_x = 0; thread_x < THREAD_SIZE_X; thread_x++){
                    accum[thread_y][thread_x] += frag_A[i%2][thread_y] * frag_B[i%2][thread_x];
                }
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
                    B_TILE_ROW_START + i + tile_idx,
                    B_TILE_COL,
                    N)]);
            }
            __syncthreads();
            write_stage_idx ^= 1;
            #pragma unroll
            for (int thread_y = 0; thread_y < THREAD_SIZE_Y; thread_y += 4) {
                FETCH_FLOAT4(frag_A[0][thread_y]) = FETCH_FLOAT4(As[load_stage_idx^1][0][THREAD_SIZE_Y * ty + thread_y]);
            }
            // load B from shared memory to register
            #pragma unroll
            for (int thread_x = 0; thread_x < THREAD_SIZE_X; thread_x += 4) {
                FETCH_FLOAT4(frag_B[0][thread_x]) = FETCH_FLOAT4(Bs[load_stage_idx^1][0][THREAD_SIZE_X * tx + thread_x]);
            }
        }
        
        #pragma unroll
        for(int thread_y = 0; thread_y < THREAD_SIZE_Y; thread_y++){
            #pragma unroll
            for(int thread_x = 0; thread_x < THREAD_SIZE_X; thread_x++){
                accum[thread_y][thread_x] += frag_A[1][thread_y] * frag_B[1][thread_x];
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
