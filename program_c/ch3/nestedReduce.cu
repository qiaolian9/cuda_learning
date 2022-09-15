#include<stdio.h>
#include<cuda_runtime.h>
#include "./func/metric.h"
#define M(x,n){x = (int*)malloc(n);}

int main(int argc, char **argv){
    printf("%s starting...\n",argv[0]);
    int dev = 3;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp,dev);
    printf("Using Device %d : %s\n",dev,deviceProp.name);

    // initial Data
    int n = 1 << 5;
    size_t nBytes = n * sizeof(int);
    int *h_idata, *h_odata, *tmp;
    M(h_idata,nBytes);
    initialData(h_idata,n);
    

    return 0;
}