#include<bits/stdc++.h>
#include<sys/time.h>
using namespace std;

template<typename T>
void checkResults(T *cpuR, T *gpuR, const int n){
    bool match = 1;
    for(int i=0;i<n;i++){
        if(cpuR[i]!=gpuR[i]){
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
        ip[i] = (T)(rand() & 0xFF);
    }
}

double cpuMsecond(){
    struct timeval t;
    gettimeofday(&t,NULL);
    return ((double)t.tv_sec*10e3 + (double)t.tv_usec * 1.0E-3);
}
