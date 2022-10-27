#include<bits/stdc++.h>
#include<sys/time.h>
using namespace std;

template<typename T>
void checkResults(T *cublasR, T *gpuR, const int m, const int n){
    bool match = 1;
    for(int i=0;i<m * n;i++){ 
        int row = i / n;
        int col = i % n;
        if(fabs(cublasR[col * m + row] - gpuR[i])>1.0e-6){
            match = false;
            printf("Don't match. Error at index %d\n",i);
            break;
        }
    }
    if(match) printf("result match...\n");
}

template<typename T>
void initialData(T *__restrict__ ip, const int n){
    time_t t;
    srand((unsigned) time(&t));
    for(int i=0;i<n;i++){
        ip[i] = i % 7;
    }
}

double cpuMsecond(){
    struct timeval t;
    gettimeofday(&t,NULL);
    return ((double)t.tv_sec*10e3 + (double)t.tv_usec * 1.0E-3);
}
