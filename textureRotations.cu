#include <stdio.h>
#include <stdint.h>

/*
    version is a hardcoded number on compile time to select rotation version.
    0 is bedrock
    1 is legacy 1.12-
    2 is 1.13 - 1.21.1
    3 is 1.21.2+
*/

#define VERSION 1


struct Pos3d{
    int32_t x;
    int32_t y;
    int32_t z;
    uint8_t rotation;
};

__device__ static inline int32_t positionalRandom(const int32_t x, const int32_t y, const int32_t z)
{
    int64_t i = (int64_t)(int32_t)(3129871ULL * (uint32_t)x) ^ (int64_t)((uint64_t)z * 116129781ULL) ^ (int64_t)y;
    i = i * i * 42317861ULL + i * 11ULL;
    return i;
}

//bedrock
__device__ static inline int32_t bedrockRotation(const int32_t x, const int32_t y, const int32_t z)
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

//1.12-
__device__ static inline int32_t getLegacyRotation(const int32_t x, const int32_t y, const int32_t z)
{
    int64_t l = (static_cast<int64_t>(x * 3129871)) ^
                (static_cast<int64_t>(z) * 116129781L) ^ static_cast<int64_t>(y);
    l = l * l * 42317861L + l * 11L;
    l = (int)l >> 16;
    return abs(l) % 4;
}

// 1.13 - 1.21.1
__device__ static inline int32_t get113Rotation(const int32_t x, const int32_t y, const int32_t z)
{
    constexpr int64_t multiplier = 0x5DEECE66DLL;
    constexpr int64_t mask = (1LL << 48) - 1;
    int64_t l = (static_cast<int64_t>(x * 3129871)) ^
                (static_cast<int64_t>(z) * 116129781L) ^ static_cast<int64_t>(y);
    l = l * l * 42317861L + l * 11L;
    int64_t seed = l >> 16;
    seed = (seed ^ multiplier) & mask;
    const int32_t rand = static_cast<int32_t>((seed * 0xBB20B4600A69L + 0x40942DE6BAL) >> 16);
    return abs(rand) % 4;
}

// 1.21.2+
__device__ static inline int32_t get1212Rotation(const int32_t x, const int32_t y, const int32_t z)
{
    constexpr int64_t multiplier = 0x5DEECE66DLL;
    constexpr int64_t mask = (1LL << 48) - 1;
    int64_t l = (static_cast<int64_t>(x * 3129871)) ^
                (static_cast<int64_t>(z) * 116129781L) ^ static_cast<int64_t>(y);
    l = l * l * 42317861L + l * 11L;
    int64_t seed = l >> 16;
    seed = (seed ^ multiplier);
    seed = seed * multiplier + 11LL & mask;
    const int32_t next = static_cast<int32_t>(seed >> (48 - 31));
    return static_cast<int32_t>((4 * static_cast<int64_t>(next)) >> 31);
}


__device__ static inline int32_t getRotation(const int32_t x, const int32_t y, const int32_t z)
{
    switch(VERSION)
    {
        case 0: //bedrock
            return bedrockRotation(x, y, z);
        case 1: //1.12- legacy
            return getLegacyRotation(x, y, z);
        case 2: //1.13 - 1.21.1 legacy
            return get113Rotation(x, y, z);
        case 3: //1.21.1+
            return get1212Rotation(x, y, z);
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

__device__ static inline uint32_t isMatching(struct Pos3d* formation, uint32_t formationCount, int32_t x, int32_t y, int32_t z)
{
    /*
    takes in a x,y,z position and returns 0 if it dosent match or 1 if it does
    */

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

    uint32_t formationCount = 25;
    struct Pos3d formation[] = {
        {0, 0, 0, 0},
        {1, 0, 0, 0},
        {2, 0, 0, 0},
        {3, 0, 0, 0},
        {4, 0, 0, 0},

        {0, 0, 1, 0},
        {1, 0, 1, 0},
        {2, 0, 1, 0},
        {3, 0, 1, 0},
        {4, 0, 1, 0},

        {0, 0, 2, 0},
        {1, 0, 2, 0},
        {2, 0, 2, 0},
        {3, 0, 2, 0},
        {4, 0, 2, 0},

        {0, 0, 3, 0},
        {1, 0, 3, 0},
        {2, 0, 3, 0},
        {3, 0, 3, 0},
        {4, 0, 3, 0},

        {0, 0, 4, 0},
        {1, 0, 4, 0},
        {2, 0, 4, 0},
        {3, 0, 4, 0},
        {4, 0, 4, 0},
    };

    //do this for all rotations
    for(int32_t i = 0; i < 4; i++) {

        //"< (xRange*zRange)"" because the bottomright most item is just the amount of positions to check
        for(int64_t position = threadId; position < ((uint64_t)xRange*(uint64_t)zRange); position += ( blockDim.x * gridDim.x ))
        {
            xPos = position%xRange + xMin;
            zPos = position/xRange + zMin;

            if(isMatching(formation, formationCount, xPos, yPos, zPos)) printf("%d,%d,%d\n", xPos, yPos, zPos);
        }
        rotate90DegCW(formation, formationCount);
    }
}

int main()
{
    cudaError_t err;

    int32_t xMin = -100000;
    int32_t zMin = -100000;

    int32_t xMax = 100000;
    int32_t zMax = 100000;

    int32_t yMin = 0;
    int32_t yMax = 0;

    for(; yMin <= yMax; yMin++)
    {
        //if you get
        //Error: too many resources requested for launch
        //decrease these parameters (the 1024s)
        spawnThread<<<1024,256>>>(xMin, xMax, zMin, zMax, yMin);

        //error checking
        err = cudaGetLastError();
        if(err != cudaSuccess){printf("Error: %s\n", cudaGetErrorString(err));exit(-1);}
        cudaDeviceSynchronize();
        //printf("complete with y=%d\n", yMin);
    }
    printf("complete\n");
}
