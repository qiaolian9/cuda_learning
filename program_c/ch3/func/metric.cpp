#include<bits/stdc++.h>
#include<sys/time.h>
using namespace std;


double cpuMSecond(){
    struct timeval t;
    gettimeofday(&t,NULL);
    return ((double)t.tv_sec * 1.0E3 + (double)t.tv_usec * 1.0E-3);
}

void initialData(int *ip, const unsigned int n){
    time_t t;
    srand((unsigned) time(&t));
    for(unsigned int i=0;i<n;i++){
        // ip[i] = (int)(rand() & 0xFF);
        ip[i] = 1;
    }
}

void checkResults(int cpu_sum, int gpu_sum){
    if(cpu_sum == gpu_sum){
        printf("Results match !!!\n");
        return ;
    }
    printf("Error...\n");
}

