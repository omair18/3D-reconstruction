#include <stdio.h>

__global__ void print_elements_kernel(unsigned char* ptr, int elementsCount)
{
    for(int i = 0; i < elementsCount; i++)
        printf("%d ", ptr[i]);
    printf("\n\0");
}

void print_elements_api(int blocks, int threads, int elementsCount, unsigned char* devPtr)
{
    print_elements_kernel<<<blocks, threads>>>(devPtr, elementsCount);
}