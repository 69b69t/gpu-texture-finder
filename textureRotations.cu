#include <stdio.h>
#include <stdint.h>

__device__ static inline int32_t random(long seed) {
    seed = (seed ^ 0x5DEECE66DL) & ((1L << 48) - 1);
    return (int)((seed * 0xBB20B4600A69L + 0x40942DE6BAL) >> 16);
}

__device__ static inline int32_t getRotation(const int32_t x, const int32_t y, const int32_t z)
{
    int64_t i = (int64_t)(int32_t)(3129871U * (uint32_t)x) ^ (int64_t)((uint64_t)z * 116129781ULL) ^ (int64_t)y;
    i = i * i * 42317861ULL + i * 11ULL;
    i = i >> 16;
    return abs(random(i)) % 4;
}

__device__ static inline int32_t isMatching(int32_t x, int32_t y, int32_t z)
{
    //hardcode rotation values. right now its set to find squares of 5x5 blocks
    //with the same texture rotations. (actually its just a single direction but too lazy to change)

    for(uint32_t j = 0; j < 5; j++)
    {
        for(uint32_t i = 0; i < 5; i++)
        {
            //set y value here
            if(getRotation(x+i, y, z+j) != 0) return 0;
        }
    }
    return 1;
}

//get id, that will be offset
//take a number and use it as counter
//each thread goes over a list of a bunch of positions
//where (x = counter/xRange)
//and (z = counter % zRange)
//once a thread reaches a position outside of the bounds, kill the thread
//thread will always go out of bounds on xMax

__global__ void spawnThread(const int32_t xMin, const int32_t xMax, const int32_t zMin, const int32_t zMax, const int32_t yPos)
{
    //each gpu thread will search a single band, from (x, -30000000) to (x, 30000000)
    //60 mil checks each
    const int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t xRange = xMax - xMin;
    const int32_t zRange = zMax - zMin;

    uint32_t xPos;
    uint32_t zPos;

    //< (xRange*zRange) because the bottomright most item is just the amount of positions to check
    for(int64_t position = threadId; position < ((uint64_t)xRange*(uint64_t)zRange); position += (1024*1024))
    {
        xPos = position%xRange + xMin;
        zPos = position/xRange + zMin;

        //without using the +xMin +zMin, the code always puts the searchy thing at 
        if(isMatching(xPos, yPos, zPos)) printf("%d,%d,%d\n", xPos, yPos, zPos);
        //if(zPos==0)printf("%d,%d,%d checked pos:%ld\n", xPos, yPos, zPos, position);

    }

}

int main()
{
    cudaError_t err;

    int32_t xMin = -50000;
    int32_t zMin = -50000;

    int32_t xMax = 50000;
    int32_t zMax = 50000;

    int32_t yMin = 0;
    int32_t yMax = 256;

    //spawnThread<<<1024,1024>>>(xMin, xMax, zMin, zMax, 64);
    //cudaDeviceSynchronize();
    //return 0;

    for(; yMin < yMax; yMin++)
    {
        spawnThread<<<1024,1024>>>(xMin, xMax, zMin, zMax, yMin);

        //error checking
        err = cudaGetLastError();
        if(err != cudaSuccess){printf("Error: %s\n", cudaGetErrorString(err));exit(-1);}
        printf("complete with y=%d\n", yMin);
        cudaDeviceSynchronize();
    }
    printf("complete\n");
}
