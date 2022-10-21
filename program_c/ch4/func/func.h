#include<stdio.h>
#include<stdlib.h>

template<typename T>
void initData(T *__restrict__ ptr, unsigned int n){
    for(int i = 0 ; i < n ; i++){
        ptr[i] = i % 13;
    }
}

template<typename T>
void resCheck(T *__restrict__ hostC, T *__restrict__ devC, unsigned int n){
    bool ck = true;
    for(int i = 0 ; i < n ; i++){
        if(abs(devC[i] - hostC[i]) > 1e-6){
            ck = false;
            printf("Error in result with index[%d]...\n", i);
            break;
        }
    }
    if(ck) printf("Result checked...\n");
}