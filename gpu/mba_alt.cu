#include <iostream>
#include <fstream>
#include <unordered_set>
#include <vector>
#include <algorithm>
#include <thrust/copy.h>
#include <thrust/remove.h>
#include <thrust/device_ptr.h>
#include <warpcore/hash_set.cuh>
#include "json.hpp"

// #define MEASUREMENT_MODE

using namespace std;
using json = nlohmann::json;

const int maxNumOfSamples = 100;
int numVar;

__constant__ uint32_t d_outputData[maxNumOfSamples];

inline
cudaError_t checkCuda(cudaError_t res) {
#ifndef MEASUREMENT_MODE
    if (res != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(res));
        assert(res == cudaSuccess);
    }
#endif
    return res;
}

tuple<uint32_t*, uint32_t*, int> readJsonFile(const std::string& filename) {

    std::ifstream file(filename);
    if (!file) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return { nullptr, nullptr, 0 };
    }

    json j;
    file >> j;

    numVar = j["initial"]["inputs"].size();
    const int numOfSamples = j["sampling"].size();

    uint32_t* inputData = new uint32_t[numOfSamples * numVar];
    uint32_t* outputData = new uint32_t[numOfSamples];

    int idx = 0;
    for (const auto& example : j["sampling"]) {
        outputData[idx / numVar] = std::stoul(example["outputs"]["0"]["value"].get<std::string>(), nullptr, 16);
        for (const auto& value : example["inputs"]) {
            inputData[idx++] = std::stoul(value["value"].get<std::string>(), nullptr, 16);
        }
    }

    return { inputData, outputData, numOfSamples };
}

__device__ uint32_t simpleHash(uint32_t input) {
    input ^= input >> 16;
    input *= 0x85ebca6b;
    input ^= input >> 13;
    input *= 0xc2b2ae35;
    input ^= input >> 16;
    return input;
}

__device__ void makeUnqChk(
    uint32_t* CS,
    int idx,
    uint64_t& seed,
    const int numOfSamples)
{

    for (int i = 0; i < numOfSamples; ++i) {
        uint64_t val = simpleHash(CS[idx * numOfSamples + i]) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        val = (i & 1) ? (val >> 32 | val << 32) : val;
        seed ^= val;
    }

}

template<class hash_set_t>
__global__ void hashSetInit(
    const int numOfSamples,
    hash_set_t hashSet,
    uint32_t* d_MBACache)
{

    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t CS[maxNumOfSamples];

    for (int i = 0; i < numOfSamples; ++i) {
        CS[i] = d_MBACache[tid * numOfSamples + i];
    }

    uint64_t seed{};
    makeUnqChk(CS, 0, seed, numOfSamples);

    const auto group = warpcore::cg::tiled_partition<1>(warpcore::cg::this_thread_block());
    hashSet.insert(seed, group);

}

enum class Op { Not, And, Or, Xor, LShift, RShift, Neg, Plus, Minus, Mul, Div, Mod };
const int numOfUnaries = 2;
const int numOfBinaries = 15; // Non-commutative operators count twice

__device__ void applyUnOp(
    uint32_t* CS,
    uint32_t* d_MBACache,
    int ldx,
    const int numOfSamples)
{

    for (int i = 0; i < numOfSamples; ++i) {
        uint32_t res = d_MBACache[ldx * numOfSamples + i];
        CS[i] = ~res; CS[i + numOfSamples] = -res;
    }

}

__device__ void applyBinOp(
    uint32_t* CS,
    uint32_t* d_MBACache,
    int ldx, int rdx,
    const int numOfSamples)
{
    bool lNoZeros = true;
    bool rNoZeros = true;
    for (int i = 0; i < numOfSamples; ++i) {
        uint32_t lRes = d_MBACache[ldx * numOfSamples + i];
        uint32_t rRes = d_MBACache[rdx * numOfSamples + i];
        lNoZeros &= lRes != 0; rNoZeros &= rRes != 0;
        CS[i] = lRes & rRes; CS[i + numOfSamples] = lRes | rRes;
        CS[i + 2 * numOfSamples] = lRes ^ rRes; CS[i + 3 * numOfSamples] = lRes << rRes;
        CS[i + 4 * numOfSamples] = rRes << lRes; CS[i + 5 * numOfSamples] = lRes >> rRes;
        CS[i + 6 * numOfSamples] = rRes >> lRes; CS[i + 7 * numOfSamples] = lRes + rRes;
        CS[i + 8 * numOfSamples] = lRes - rRes; CS[i + 9 * numOfSamples] = rRes - lRes;
        CS[i + 10 * numOfSamples] = lRes * rRes; CS[i + 11 * numOfSamples] = lRes / rRes;
        CS[i + 12 * numOfSamples] = rRes / lRes; CS[i + 13 * numOfSamples] = lRes % rRes;
        CS[i + 14 * numOfSamples] = rRes % lRes;
    }
    if (!lNoZeros) {
        for (int i = 0; i < numOfSamples; ++i) {
            uint32_t rRes = d_MBACache[rdx * numOfSamples + i];
            CS[i + 12 * numOfSamples] = rRes; CS[i + 14 * numOfSamples] = rRes;
        }
    }
    if (!rNoZeros) {
        for (int i = 0; i < numOfSamples; ++i) {
            uint32_t lRes = d_MBACache[ldx * numOfSamples + i];
            CS[i + 11 * numOfSamples] = lRes; CS[i + 13 * numOfSamples] = lRes;
        }
    }
}

template<class hash_set_t>
__device__ bool processUniqueCS(
    uint32_t* CS, bool* d_CSUnq,
    const int numOfSamples, const int numOfFormulas,
    hash_set_t& hashSet)
{

    const auto group = warpcore::cg::tiled_partition<1>(warpcore::cg::this_thread_block());
    uint64_t seed;

    for (int i = 0; i < numOfFormulas; ++i) {
        seed = 0;
        makeUnqChk(CS, i, seed, numOfSamples);
        d_CSUnq[i] = (hashSet.insert(seed, group) > 0) ? false : true;
    }

    return d_CSUnq;

}

__device__ void insertInCache(
    uint32_t* CS,
    bool* d_CSUnq,
    int tid, int ldx, int rdx,
    const int numOfSamples, const int numOfFormulas,
    uint32_t* d_temp_MBACache, bool* d_temp_boolCache,
    int* d_temp_leftIdx, int* d_temp_rightIdx, int* d_temp_opIdx, int* d_opIdxs,
    int* d_FinalMBAIdx)
{

    int modTid = tid * numOfFormulas;
    bool found;

    for (int j = 0; j < numOfFormulas; ++j) {

        if (d_CSUnq[j]) {

            for (int i = 0; i < numOfSamples; ++i) {
                d_temp_MBACache[modTid * numOfSamples + i] = CS[j * numOfSamples + i];
                d_temp_boolCache[modTid * numOfSamples + i] = true;
            }
            d_temp_leftIdx[modTid] = ldx; d_temp_rightIdx[modTid] = rdx;
            d_temp_opIdx[modTid] = d_opIdxs[j];

            found = true;
            for (int i = 0; found && i < numOfSamples; ++i) found = (CS[j * numOfSamples + i] == d_outputData[i]);
            if (found) atomicCAS(d_FinalMBAIdx, -1, modTid);

        } else {

            for (int i = 0; i < numOfSamples; ++i) {
                d_temp_boolCache[modTid * numOfSamples + i] = false;
            }
            d_temp_leftIdx[modTid] = -1; d_temp_rightIdx[modTid] = -1;
            d_temp_opIdx[modTid] = -1;

        }

        modTid++;

    }

}

template<class hash_set_t>
__global__ void processUnaries(
    const int idx1, const int idx2,
    const int numOfSamples,
    uint32_t* d_MBACache, uint32_t* d_temp_MBACache, bool* d_temp_boolCache,
    int* d_temp_leftIdx, int* d_temp_rightIdx, int* d_temp_opIdx, int* d_unOpIdxs,
    int* d_FinalMBAIdx,
    hash_set_t hashSet)
{

    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int maxTid = idx2 - idx1 + 1;

    if (tid < maxTid) {

        int ldx = idx1 + tid;
        uint32_t CS[maxNumOfSamples * numOfUnaries];
        bool d_CSUnq[numOfUnaries];

        applyUnOp(CS, d_MBACache, ldx, numOfSamples);
        processUniqueCS(CS, d_CSUnq, numOfSamples, numOfUnaries, hashSet);
        insertInCache(
            CS, d_CSUnq, tid, ldx, 0, numOfSamples, numOfUnaries, d_temp_MBACache, d_temp_boolCache,
            d_temp_leftIdx, d_temp_rightIdx, d_temp_opIdx, d_unOpIdxs, d_FinalMBAIdx);

    }

}

template<class hash_set_t>
__global__ void processBinaries(
    const int idx1, const int idx2, const int idx3, const int idx4,
    const int numOfSamples,
    uint32_t* d_MBACache, uint32_t* d_temp_MBACache, bool* d_temp_boolCache,
    int* d_temp_leftIdx, int* d_temp_rightIdx, int* d_temp_opIdx, int* d_binOpIdxs,
    int* d_FinalMBAIdx,
    hash_set_t hashSet)
{

    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int maxTid = (idx4 - idx3 + 1) * (idx2 - idx1 + 1);

    if (tid < maxTid) {

        int ldx = idx1 + tid / (idx4 - idx3 + 1);
        int rdx = idx3 + tid % (idx4 - idx3 + 1);
        uint32_t CS[maxNumOfSamples * numOfBinaries];
        bool d_CSUnq[numOfBinaries];

        applyBinOp(CS, d_MBACache, ldx, rdx, numOfSamples);
        processUniqueCS(CS, d_CSUnq, numOfSamples, numOfBinaries, hashSet);
        insertInCache(
            CS, d_CSUnq, tid, ldx, rdx, numOfSamples, numOfBinaries, d_temp_MBACache, d_temp_boolCache,
            d_temp_leftIdx, d_temp_rightIdx, d_temp_opIdx, d_binOpIdxs, d_FinalMBAIdx);

    }

}

// Transfering the unique CSs from temporary cache to MBACache
bool storeUniqueMBAs(
    int N,
    int& lastIdx,
    const int numOfSamples,
    const int MBACacheCapacity,
    uint32_t* d_MBACache, uint32_t* d_temp_MBACache, bool* d_temp_boolCache,
    int* d_leftIdx, int* d_rightIdx, int* d_opIdx,
    int* d_temp_leftIdx, int* d_temp_rightIdx, int* d_temp_opIdx)
{

    thrust::device_ptr<uint32_t> new_end_ptr;
    thrust::device_ptr<uint32_t> d_MBACache_ptr(d_MBACache + lastIdx * numOfSamples);
    thrust::device_ptr<uint32_t> d_temp_MBACache_ptr(d_temp_MBACache);
    thrust::device_ptr<bool> d_temp_boolCache_ptr(d_temp_boolCache);
    thrust::device_ptr<int> d_leftIdx_ptr(d_leftIdx + lastIdx);
    thrust::device_ptr<int> d_rightIdx_ptr(d_rightIdx + lastIdx);
    thrust::device_ptr<int> d_opIdx_ptr(d_opIdx + lastIdx);
    thrust::device_ptr<int> d_temp_leftIdx_ptr(d_temp_leftIdx);
    thrust::device_ptr<int> d_temp_rightIdx_ptr(d_temp_rightIdx);
    thrust::device_ptr<int> d_temp_opIdx_ptr(d_temp_opIdx);

    new_end_ptr = thrust::remove_if(
        d_temp_MBACache_ptr, d_temp_MBACache_ptr + N * numOfSamples,
        d_temp_boolCache_ptr, thrust::logical_not<bool>());
    thrust::remove(d_temp_leftIdx_ptr, d_temp_leftIdx_ptr + N, -1);
    thrust::remove(d_temp_rightIdx_ptr, d_temp_rightIdx_ptr + N, -1);
    thrust::remove(d_temp_opIdx_ptr, d_temp_opIdx_ptr + N, -1);

    int numberOfNewUniqueMBAs = static_cast<int>(new_end_ptr - d_temp_MBACache_ptr) / numOfSamples;
    if (lastIdx + numberOfNewUniqueMBAs > MBACacheCapacity) {
        return true;
    } else {
        N = numberOfNewUniqueMBAs;
        thrust::copy_n(d_temp_MBACache_ptr, N * numOfSamples, d_MBACache_ptr);
        thrust::copy_n(d_temp_leftIdx_ptr, N, d_leftIdx_ptr);
        thrust::copy_n(d_temp_rightIdx_ptr, N, d_rightIdx_ptr);
        thrust::copy_n(d_temp_opIdx_ptr, N, d_opIdx_ptr);
        lastIdx += N;
        return false;
    }

}

// Finding the left and right indices that makes the final MBA to bring to the host later
__global__ void generateResIndices(
    const int numVar,
    const int index,
    const int* d_leftIdx, const int* d_rightIdx, const int* d_opIdx,
    int* d_resIndices)
{

    int resIdx = 0;
    while (d_resIndices[resIdx] != -1) resIdx++;
    int queue[100];
    queue[0] = index;
    int head = 0;
    int tail = 1;
    while (head < tail) {
        int mba = queue[head];
        int l = d_leftIdx[mba];
        int r = d_rightIdx[mba];
        int op = d_opIdx[mba];
        d_resIndices[resIdx++] = mba;
        d_resIndices[resIdx++] = l;
        d_resIndices[resIdx++] = r;
        d_resIndices[resIdx++] = op;
        if (l >= numVar) queue[tail++] = l;
        if (r >= numVar) queue[tail++] = r;
        head++;
    }

}

const string opStr[12] = { "~", "&", "|", "^", "<<", ">>", "-", "+", "-", "*", "/", "%" };

// Generating the final RE string recursively
// When all the left and right indices are ready in the host
string toString(int index, map<int, tuple<int, int, int>>& indicesMap)
{

    if (index < numVar) { return "v" + to_string(index); }

    auto [l, r, op] = indicesMap[index];
    bool isUnary = (op == 0 | op == 6);

    if (isUnary) {
        string res = toString(l, indicesMap);
        return opStr[op] + "(" + res + ")";
    }

    else {
        string left = toString(l, indicesMap);
        string right = toString(r, indicesMap);
        return "(" + left + ")" + opStr[op] + "(" + right + ")";
    }

}

// Bringing the left and right indices of the MBA from device to host
string MBAToString(
    const int FinalMBAIdx,
    const int* d_leftIdx, const int* d_rightIdx, const int* d_opIdx,
    const int* d_temp_leftIdx, const int* d_temp_rightIdx, const int* d_temp_opIdx)
{

    int* LIdx = new int[1];
    int* RIdx = new int[1];
    int* OIdx = new int[1];

    checkCuda(cudaMemcpy(LIdx, d_temp_leftIdx + FinalMBAIdx, sizeof(int), cudaMemcpyDeviceToHost));
    checkCuda(cudaMemcpy(RIdx, d_temp_rightIdx + FinalMBAIdx, sizeof(int), cudaMemcpyDeviceToHost));
    checkCuda(cudaMemcpy(OIdx, d_temp_opIdx + FinalMBAIdx, sizeof(int), cudaMemcpyDeviceToHost));

    int* d_resIndices;
    checkCuda(cudaMalloc(&d_resIndices, 100 * sizeof(int)));

    thrust::device_ptr<int> d_resIndices_ptr(d_resIndices);
    thrust::fill(d_resIndices_ptr, d_resIndices_ptr + 100, -1);

    if (*LIdx >= numVar) generateResIndices << <1, 1 >> > (numVar, *LIdx, d_leftIdx, d_rightIdx, d_opIdx, d_resIndices);
    if (*RIdx >= numVar) generateResIndices << <1, 1 >> > (numVar, *RIdx, d_leftIdx, d_rightIdx, d_opIdx, d_resIndices);

    int resIndices[100];
    checkCuda(cudaMemcpy(resIndices, d_resIndices, 100 * sizeof(int), cudaMemcpyDeviceToHost));

    map<int, tuple<int, int, int>> indicesMap;
    indicesMap.insert(make_pair(INT_MAX, make_tuple(*LIdx, *RIdx, *OIdx)));
    cudaFree(d_resIndices);

    int i = 0;
    while (resIndices[i] != -1 && i + 2 < 100) {
        int mba = resIndices[i];
        int l = resIndices[i + 1];
        int r = resIndices[i + 2];
        int op = resIndices[i + 3];
        indicesMap.insert(make_pair(mba, make_tuple(l, r, op)));
        i += 4;
    }

    if (i + 3 >= 100) return "Size of the output is too big !";
    else return toString(INT_MAX, indicesMap);

}

template<Op op, class hash_set_t>
bool generateUnaryMBAs(
    int MBALen,
    int* startPoints,
    const int temp_MBACacheCapacity, const int MBACacheCapacity,
    uint64_t& allMBAs, int& lastIdx,
    const int numOfSamples,
    uint32_t* d_MBACache, uint32_t* d_temp_MBACache, bool* d_temp_boolCache,
    int* d_temp_leftIdx, int* d_temp_rightIdx, int* d_temp_opIdx,
    int* d_leftIdx, int* d_rightIdx, int* d_opIdx,
    int* d_unOpIdxs,
    int* d_FinalMBAIdx, int* FinalMBAIdx,
    hash_set_t hashSet)
{

    bool lastRound = false;
    int idx1 = startPoints[MBALen - 1];
    int idx2 = startPoints[MBALen] - 1;
    int N = idx2 - idx1 + 1;

    if (N) {
        int x = idx1, y;
        do {
            y = x + min(temp_MBACacheCapacity / numOfUnaries - 1, idx2 - x);
            N = y - x + 1;
#ifndef MEASUREMENT_MODE
            printf("Length %-2d | UnOps  | AllMBAs: %-11lu | StoredMBAs: %-10d | ToBeChecked: %-10d \n",
                MBALen, allMBAs, lastIdx, numOfUnaries * N);
#endif
            int Blc = (N + 1023) / 1024;
            processUnaries<hash_set_t> << <Blc, 1024 >> > (
                x, y, numOfSamples, d_MBACache, d_temp_MBACache, d_temp_boolCache,
                d_temp_leftIdx, d_temp_rightIdx, d_temp_opIdx, d_unOpIdxs, d_FinalMBAIdx, hashSet);
            checkCuda(cudaPeekAtLastError());
            checkCuda(cudaMemcpy(FinalMBAIdx, d_FinalMBAIdx, sizeof(int), cudaMemcpyDeviceToHost));
            allMBAs += numOfUnaries * N;
            if (*FinalMBAIdx != -1) return true;
            lastRound = storeUniqueMBAs(
                numOfUnaries * N, lastIdx, numOfSamples, MBACacheCapacity, d_MBACache, d_temp_MBACache,
                d_temp_boolCache, d_leftIdx, d_rightIdx, d_opIdx, d_temp_leftIdx, d_temp_rightIdx, d_temp_opIdx);
            if (lastRound) return true;
            x = y + 1;
        } while (y < idx2);
    }

    return false;

}

template<Op op, class hash_set_t>
bool generateBinaryMBAs(
    int MBALen,
    int* startPoints,
    const int temp_MBACacheCapacity, const int MBACacheCapacity,
    uint64_t& allMBAs, int& lastIdx,
    const int numOfSamples,
    uint32_t* d_MBACache, uint32_t* d_temp_MBACache, bool* d_temp_boolCache,
    int* d_temp_leftIdx, int* d_temp_rightIdx, int* d_temp_opIdx,
    int* d_leftIdx, int* d_rightIdx, int* d_opIdx,
    int* d_binOpIdxs,
    int* d_FinalMBAIdx, int* FinalMBAIdx,
    hash_set_t hashSet)
{

    bool lastRound = false;

    for (int i = 1; 2 * i <= MBALen - 1; ++i) {

        int idx1 = startPoints[i];
        int idx2 = startPoints[i + 1] - 1;
        int idx3 = startPoints[MBALen - i - 1];
        int idx4 = startPoints[MBALen - i] - 1;
        int N = (idx4 - idx3 + 1) * (idx2 - idx1 + 1);

        if (N) {
            int x = idx3, y;
            do {
                y = x + min(temp_MBACacheCapacity / (numOfBinaries * (idx2 - idx1 + 1)) - 1, idx4 - x);
                N = (y - x + 1) * (idx2 - idx1 + 1);
#ifndef MEASUREMENT_MODE
                printf("Length %-2d | BinOps | AllMBAs: %-11lu | StoredMBAs: %-10d | ToBeChecked: %-10d \n",
                    MBALen, allMBAs, lastIdx, numOfBinaries * N);
#endif
                int Blc = (N + 1023) / 1024;
                processBinaries<hash_set_t> << <Blc, 1024 >> > (
                    idx1, idx2, x, y, numOfSamples, d_MBACache, d_temp_MBACache, d_temp_boolCache,
                    d_temp_leftIdx, d_temp_rightIdx, d_temp_opIdx, d_binOpIdxs, d_FinalMBAIdx, hashSet);
                checkCuda(cudaPeekAtLastError());
                checkCuda(cudaMemcpy(FinalMBAIdx, d_FinalMBAIdx, sizeof(int), cudaMemcpyDeviceToHost));
                allMBAs += numOfBinaries * N;
                if (*FinalMBAIdx != -1) return true;
                lastRound = storeUniqueMBAs(
                    numOfBinaries * N, lastIdx, numOfSamples, MBACacheCapacity, d_MBACache, d_temp_MBACache,
                    d_temp_boolCache, d_leftIdx, d_rightIdx, d_opIdx, d_temp_leftIdx, d_temp_rightIdx, d_temp_opIdx);
                if (lastRound) return true;
                x = y + 1;
            } while (y < idx4);
        }

    }

    startPoints[MBALen + 1] = lastIdx;
    return false;

}

string MBA(
    const int maxLen,
    const int numOfSamples,
    uint32_t* inputData,
    uint32_t* outputData)
{

    // ---------------------------------
    // Generating and checking variables
    // ---------------------------------

    if (numOfSamples > maxNumOfSamples) {
        printf("This version supports at most %d samples.\n", maxNumOfSamples);
        return "see_the_error";
    }

    // Copying outputs into the constant memory
    checkCuda(cudaMemcpyToSymbol(d_outputData, outputData, numOfSamples * sizeof(uint32_t)));

    // Creating the cache
    uint32_t* MBACache = new uint32_t[numVar * numOfSamples];

    // Creating operators indexes
    int unOpIdxs[] = { 0, 6 };
    int* d_unOpIdxs;
    checkCuda(cudaMalloc(&d_unOpIdxs, numOfUnaries * sizeof(int)));
    checkCuda(cudaMemcpy(d_unOpIdxs, unOpIdxs, numOfUnaries * sizeof(int), cudaMemcpyHostToDevice));

    int binOpIdxs[] = { 1, 2, 3, 4, 4, 5, 5, 7, 8, 8, 9, 10, 10, 11, 11 };
    int* d_binOpIdxs;
    checkCuda(cudaMalloc(&d_binOpIdxs, numOfBinaries * sizeof(int)));
    checkCuda(cudaMemcpy(d_binOpIdxs, binOpIdxs, numOfBinaries * sizeof(int), cudaMemcpyHostToDevice));

    // Number of generated formulas
    uint64_t allMBAs{};

    // Index of the last free position in the cache
    int lastIdx{};

#ifndef MEASUREMENT_MODE
    printf("Length %-2d | Vars   | AllMBAs: %-11lu | StoredMBAs: %-10d | ToBeChecked: %-10d \n",
        1, allMBAs, 0, numVar);
#endif

    // Initializing the cache with variables
    int index{};
    for (int i = 0; i < numVar; ++i) {
        bool found = true;
        for (int j = 0; j < numOfSamples; j++) {
            MBACache[index++] = inputData[j * numVar + i];
            if (!(inputData[j * numVar + i] == outputData[j])) found = false;
        }
        allMBAs++; lastIdx++;
        if (found) return "v" + to_string(i);
    }

    // ------------------------------------
    // Memory allocation and initialisation
    // ------------------------------------

    int maxAllocationSize;
    cudaDeviceGetAttribute(&maxAllocationSize, cudaDevAttrMaxPitch, 0);

    const int MBACacheCapacity = maxAllocationSize / (numOfSamples * sizeof(uint32_t));
    const int temp_MBACacheCapacity = MBACacheCapacity / 2;

    // Unary operators : ~, Neg
    // Binary operators : &, |, ^, <<, >>, +, -, *, /, %
    int* startPoints = new int[maxLen + 2]();
    startPoints[1] = 0;
    startPoints[2] = lastIdx;

    int* d_FinalMBAIdx;
    int* FinalMBAIdx = new int[1]; *FinalMBAIdx = -1;
    checkCuda(cudaMalloc(&d_FinalMBAIdx, sizeof(int)));
    checkCuda(cudaMemcpy(d_FinalMBAIdx, FinalMBAIdx, sizeof(int), cudaMemcpyHostToDevice));

    uint32_t* d_MBACache, * d_temp_MBACache;
    bool* d_temp_boolCache;
    int* d_leftIdx, * d_rightIdx, * d_opIdx, * d_temp_leftIdx, * d_temp_rightIdx, * d_temp_opIdx;
    checkCuda(cudaMalloc(&d_leftIdx, MBACacheCapacity * sizeof(int)));
    checkCuda(cudaMalloc(&d_rightIdx, MBACacheCapacity * sizeof(int)));
    checkCuda(cudaMalloc(&d_opIdx, MBACacheCapacity * sizeof(int)));
    checkCuda(cudaMalloc(&d_temp_leftIdx, temp_MBACacheCapacity * sizeof(int)));
    checkCuda(cudaMalloc(&d_temp_rightIdx, temp_MBACacheCapacity * sizeof(int)));
    checkCuda(cudaMalloc(&d_temp_opIdx, temp_MBACacheCapacity * sizeof(int)));
    checkCuda(cudaMalloc(&d_MBACache, MBACacheCapacity * numOfSamples * sizeof(uint32_t)));
    checkCuda(cudaMalloc(&d_temp_MBACache, temp_MBACacheCapacity * numOfSamples * sizeof(uint32_t)));
    checkCuda(cudaMalloc(&d_temp_boolCache, temp_MBACacheCapacity * numOfSamples * sizeof(bool)));

    using hash_set_t = warpcore::HashSet<
        uint64_t,         // Key type
        uint64_t(0) - 1,  // Empty key
        uint64_t(0) - 2,  // Tombstone key
        warpcore::probing_schemes::QuadraticProbing<warpcore::hashers::MurmurHash <uint64_t>>>;

    hash_set_t hashSet(2 * MBACacheCapacity);

    checkCuda(cudaMemcpy(d_MBACache, MBACache, numVar * numOfSamples * sizeof(uint32_t), cudaMemcpyHostToDevice));
    hashSetInit<hash_set_t> << <1, numVar >> > (numOfSamples, hashSet, d_MBACache);

    // ----------------------------
    // Enumeration of the next MBAs
    // ----------------------------

    bool lastRound = false;

    for (int MBALen = 2; MBALen <= maxLen; ++MBALen) {

        lastRound = generateUnaryMBAs<Op::Not>(MBALen, startPoints, temp_MBACacheCapacity, MBACacheCapacity, allMBAs, lastIdx,
            numOfSamples, d_MBACache, d_temp_MBACache, d_temp_boolCache, d_temp_leftIdx, d_temp_rightIdx, d_temp_opIdx,
            d_leftIdx, d_rightIdx, d_opIdx, d_unOpIdxs, d_FinalMBAIdx, FinalMBAIdx, hashSet);
        if (lastRound) break;

        lastRound = generateBinaryMBAs<Op::Not>(MBALen, startPoints, temp_MBACacheCapacity, MBACacheCapacity, allMBAs, lastIdx,
            numOfSamples, d_MBACache, d_temp_MBACache, d_temp_boolCache, d_temp_leftIdx, d_temp_rightIdx, d_temp_opIdx,
            d_leftIdx, d_rightIdx, d_opIdx, d_binOpIdxs, d_FinalMBAIdx, FinalMBAIdx, hashSet);
        if (lastRound) break;

    }

    string output;

    if (*FinalMBAIdx != -1) {
        output = MBAToString(*FinalMBAIdx, d_leftIdx, d_rightIdx, d_opIdx, d_temp_leftIdx, d_temp_rightIdx, d_temp_opIdx);
    } else {
        output = "Not found !";
    }

    // Cleanup
    cudaFree(d_MBACache);
    cudaFree(d_temp_MBACache);
    cudaFree(d_temp_boolCache);
    cudaFree(d_leftIdx);
    cudaFree(d_rightIdx);
    cudaFree(d_opIdx);
    cudaFree(d_temp_leftIdx);
    cudaFree(d_temp_rightIdx);
    cudaFree(d_temp_opIdx);
    cudaFree(d_FinalMBAIdx);

    return output;

}

int main(int argc, char* argv[]) {

    // -----------------
    // Reading the input
    // -----------------

    if (argc != 3) {
        printf("Arguments should be in the form of\n");
        printf("-----------------------------------------------------------------\n");
        printf("%s <input_file_address> <maxLen>\n", argv[0]);
        printf("-----------------------------------------------------------------\n");
        return 0;
    }

    if (atoi(argv[2]) < 1 || atoi(argv[2]) > 100) {
        printf("Argument maxLen = %s should be between 1 and 100", argv[2]);
        return 0;
    }

    string fileName = argv[1];
    auto [inputData, outputData, numOfSamples] = readJsonFile(fileName);
    int maxLen = atoi(argv[2]);

    // ------------------------------
    // Mixed Boolean Arithmetic (MBA)
    // ------------------------------

#ifdef MEASUREMENT_MODE
    auto start = chrono::high_resolution_clock::now();
#endif

    string output = MBA(maxLen, numOfSamples, inputData, outputData);
    if (output == "see_the_error") return 0;

#ifdef MEASUREMENT_MODE
    auto stop = chrono::high_resolution_clock::now();
#endif

    // -------------------
    // Printing the output
    // -------------------

    // TO DO

#ifdef MEASUREMENT_MODE
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
    // printf("\nNumber of All LTLs: %lu", allLTLs);
    // printf("\nCost of Final LTL: %d", LTLcost);
    printf("\nRunning Time: %f s", (double)duration * 0.000001);
#endif
    printf("\nMBA: \"%s\"\n", output.c_str());

    return 0;
}