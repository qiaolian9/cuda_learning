#include<iostream>
#include<sys/time.h>
#include<stdlib.h>

void sumArray(float *A, float *B, float *C, const int N){
    for(int i=0; i<N; i++){
        C[i] = A[i] + B[i];
    }
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

    // func_Ptr
    void (*func)(float*, float*, float*, const int);
    func = sumArray;
    
    struct timeval t1, t2;
    double timeuse = 0;

    gettimeofday(&t1,NULL);
    func(A,B,C,N);
    gettimeofday(&t2,NULL);

    timeuse = (t2.tv_sec - t1.tv_sec) * 1000.0f + (double)((t2.tv_usec - t1.tv_usec + 1000000) % 1000000) / 1000;

    std::cout << "array sum cost time is " << timeuse << "ms" << std::endl;

    free(A);
    free(B);
    free(C);
    
    return 0;
}