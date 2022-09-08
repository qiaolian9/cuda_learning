#include<stdio.h>

__global__
void myprint(void){
    int index = threadIdx.x;
    if(index == 5)
        printf("hello world gpu thread %d!\n",index);
    return ;
}

int main(){
    printf("hello world CPU!\n");
    int num_Blocks = 1;
    int num_threads = 10;
    myprint<<<num_Blocks,num_threads>>>();
    cudaDeviceSynchronize();
    // cudaDeviceReset();
    return 0;
}