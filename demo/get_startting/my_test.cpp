#include<iostream>
#include<sys/time.h>

#define T 512
#define B 10
#define C 16

void cal(int **W, int ***K, int ***x){
    for(int i=0;i<B;i++){
        for(int j=0;j<C;j++){
            for(int k=0;k<T;k++){
                x[i][j][k] = 0;
                for(int l=0;l<k;l++){
                    x[i][j][k] += W[j][l] * K[i][j][l];
                }
            }
        }
    }
    return ;
}

int main(){
    // ptr_func
    void(*func)(int**, int***, int***);
    func = cal;
    // int W[C][T], K[B][C][T], x[B][C][T];
    int **W = new int* [C];
    // init W
    for(int i = 0; i < C; i++){
        *(W+i) = new int [T];
        for(int j = 0; j < T; j++){
            W[i][j] = i * T + j;
        }
    }
    int ***K,***x;
    K = new int** [B];
    x = new int** [B];
    // init K
    for(int i = 0; i < B; i++){
        *(K+i) = new int* [C];
        *(x+i) = new int* [C];
        for(int j = 0; j < C; j++){
            *(*(x+i)+j) = new int [T];
            *(*(K+i)+j) = new int [T];
            for(int k = 0; k < T; k++){
                K[i][j][k] = i * (T * C) + j * T + k;
            }
        }
    }
    // calculate
    struct timeval t1, t2;
    double timeuse = 0;
    int n = 10;
    for(int i=0;i<n;i++){
        gettimeofday(&t1, NULL);
        // std::cout << **(W+1) << std::endl;
        func(W,K,x);
        gettimeofday(&t2, NULL);
        timeuse += (t2.tv_sec - t1.tv_sec) + (double)((t2.tv_usec - t1.tv_usec + 1000000) % 1000000) / 1000;
    }
    timeuse /= n;
    std::cout << "DWop mean(10x) cost time is " << timeuse << "ms" << std::endl;

    // free memory
    for(int i=0;i<B;i++){
        for(int j=0;j<C;j++){
            delete [] x[i][j];
            delete [] K[i][j];
        }
    }
    for(int i=0;i<B;i++){
        delete [] x[i];
        delete [] K[i];
    }
    for(int i=0;i<C;i++) delete [] W[i];
    delete [] x;
    delete [] K;
    delete [] W;
    
    return 0;
}