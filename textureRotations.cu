#include <stdio.h>
#include <stdint.h>

#define UNKNOWN_ROTATION 1

//if IS_BEDROCK flag is set, ignore the IS_112 flag
#define IS_BEDROCK 1
#define IS_112 1


struct Pos3d{
    int32_t x;
    int32_t y;
    int32_t z;
    uint8_t rotation;
};

__device__ static inline int32_t random(long seed) {
    seed = (seed ^ 0x5DEECE66DULL) & ((1ULL << 48) - 1);
    return (int)((seed * 0xBB20B4600A69ULL + 0x40942DE6BAULL) >> 16);
}

__device__ static inline int32_t getRotation(const int32_t x, const int32_t y, const int32_t z)
{
    /*
    gets the rotation at a block
    */
    if(!IS_BEDROCK)
    {
        //java stuff
        int64_t i = (int64_t)(int32_t)(3129871ULL * (uint32_t)x) ^ (int64_t)((uint64_t)z * 116129781ULL) ^ (int64_t)y;
        i = i * i * 42317861ULL + i * 11ULL;
        
        //int cast for 1.12-, otherwise none
        if(!IS_112) i = i >> 16;
        else i = (int)i >> 16;

        //no random call in 1.12-
        if(!IS_112) return abs(random(i)) % 4;
        else return abs(i) % 4;
    }
    else
    {
        //bedrock stuff. thanks andrew!
        int uVar1 = (z * 0x6ebfff5U) ^ (x * 0x2fc20fU) ^ y;
        uVar1 = (uVar1 * 0x285b825 + 0xb) * uVar1;
        int rot = (uVar1 >> 24) & 3;
        if ((rot & 2) != 0)
        {
            rot ^= 1;
        }
        return rot;
    }
}

__device__ static inline void rotate90DegCW(struct Pos3d* formation, uint32_t formationCount)
{
    /*
    rotates a formation 90 degrees clockwise
    */
    int32_t temp;
    for(uint32_t i = 0; i < formationCount; i++)
    {
        //(x,z) rotated 90 deg would be (z,-x)
        temp = formation[i].x;
        formation[i].x = formation[i].z;
        formation[i].z = temp;

        formation[i].x = -formation[i].x;

        formation[i].rotation = ((formation[i].rotation+1) % 4);
    }
}

__device__ static inline uint32_t checkFormation(struct Pos3d* formation, uint32_t formationCount, int32_t x, int32_t y, int32_t z)
{
    /*
    takes an x,y,z position and a formation and checks ONE orientation
    returns 0 if not a match, and 1 if there is
    */
    for(uint32_t i = 0; i < formationCount; i++)
    {
        //if block rotation is not equal to the rotation we're searching for, invalid
        if(getRotation(x+formation[i].x, y+formation[i].y, z+formation[i].z) != ((formation[i].rotation) % 4)) return 0;
    }
    return 1;
}

__device__ static inline uint32_t isMatching(int32_t x, int32_t y, int32_t z)
{
    /*
    takes in a x,y,z position and returns 0 if it dosent match or 1 if it does
    */


    //change these and reference the definition of Pos3d
    uint32_t formationCount = 16;


    struct Pos3d formation[] = {
        {0, 0, 0, 0},
        {1, 0, 0, 0},
        {2, 0, 0, 0},
        {3, 0, 0, 0},

        {0, 0, 1, 0},
        {1, 0, 1, 0},
        {2, 0, 1, 0},
        {3, 0, 1, 0},

        {0, 0, 2, 0},
        {1, 0, 2, 0},
        {2, 0, 2, 0},
        {3, 0, 2, 0},

        {0, 0, 3, 0},
        {1, 0, 3, 0},
        {2, 0, 3, 0},
        {3, 0, 3, 0}
    };

    //this is unreasonably 3x faster
    if(UNKNOWN_ROTATION)
    {
        if(checkFormation(formation, formationCount, x, y, z)) return 1;
        rotate90DegCW(formation, formationCount);

        if(checkFormation(formation, formationCount, x, y, z)) return 1;
        rotate90DegCW(formation, formationCount);

        if(checkFormation(formation, formationCount, x, y, z)) return 1;
        rotate90DegCW(formation, formationCount);
    }

    if(checkFormation(formation, formationCount, x, y, z)) return 1;

    //else is invalid
    return 0;
}

__global__ void spawnThread(const int32_t xMin, const int32_t xMax, const int32_t zMin, const int32_t zMax, const int32_t yPos)
{
    /*
    spawns a single searcher thread that is aware of which thread it is
    */
    const int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t xRange = xMax - xMin;
    const int32_t zRange = zMax - zMin;

    uint32_t xPos;
    uint32_t zPos;

    //"< (xRange*zRange)"" because the bottomright most item is just the amount of positions to check
    for(int64_t position = threadId; position < ((uint64_t)xRange*(uint64_t)zRange); position += ( blockDim.x * gridDim.x ))
    {
        xPos = position%xRange + xMin;
        zPos = position/xRange + zMin;

        if(isMatching(xPos, yPos, zPos)) printf("%d,%d,%d\n", xPos, yPos, zPos);
    }

}

int main()
{
    cudaError_t err;

    int32_t xMin = 0;
    int32_t zMin = 0;

    int32_t xMax = 50000;
    int32_t zMax = 50000;

    int32_t yMin = -61;
    int32_t yMax = -61;

    for(; yMin <= yMax; yMin++)
    {
        //if you get
        //Error: too many resources requested for launch
        //decrease these parameters (the 1024s)
        spawnThread<<<1024,512>>>(xMin, xMax, zMin, zMax, yMin);

        //error checking
        err = cudaGetLastError();
        if(err != cudaSuccess){printf("Error: %s\n", cudaGetErrorString(err));exit(-1);}
        cudaDeviceSynchronize();
        printf("complete with y=%d\n", yMin);
    }
    printf("complete\n");
}
