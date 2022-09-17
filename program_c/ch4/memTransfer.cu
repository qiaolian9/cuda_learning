#include<stdio.h>
#include<cuda_runtime.h>
#define M(x,n){x = (float*)malloc(n);}
#define cM(x,n){cudaMalloc((void**)&x,n);}

int main(int argc,char **argv){
    int dev = 3;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp,dev);
    printf("%s starting...\n",argv[0]);
    printf("Using Device %d : %s\n",dev,deviceProp.name);
    cudaSetDevice(dev);

    // initial data
    unsigned int n = 1<<22;
    unsigned int nBytes = n * sizeof(float);
    printf("Data size %5.2f MB\n",nBytes / (1024.0f * 1024.0f));

    // allocate the host memory
    float *h_g, *d_g;
    M(h_g,nBytes);
    cM(d_g,nBytes);

    for(unsigned int i=0;i<n;i++) h_g[i] = 1.0f;

    // transfer data
    cudaMemcpy(d_g,h_g,nBytes,cudaMemcpyHostToDevice);
    cudaMemcpy(h_g,d_g,nBytes,cudaMemcpyDeviceToHost);
    
    // free memory
    free(h_g);
    cudaFree(d_g);
    cudaDeviceReset();
    return 0;
}