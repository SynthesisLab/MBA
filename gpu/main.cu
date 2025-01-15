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

void printDeviceArray(const uint32_t* d_array, int size, const int numOfSamples) {
    uint32_t* h_array = new uint32_t[size];  // Allouer de la mémoire sur le CPU
    cudaMemcpy(h_array, d_array, size * sizeof(uint32_t), cudaMemcpyDeviceToHost);  // Copier depuis le GPU

    cout << "Array elements:\n";
    for (int i = 0; i < size / numOfSamples; ++i) {
        for (int j = 0; j < numOfSamples; ++j) {
            cout << h_array[i * numOfSamples + j] << " ";
        }
        cout << endl << endl;
    }

    delete[] h_array;  // Libérer la mémoire sur le CPU
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
    uint64_t& seed,
    const int numOfSamples)
{

    for (int i = 0; i < numOfSamples; ++i) {
        uint64_t val = simpleHash(CS[i]) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
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
    makeUnqChk(CS, seed, numOfSamples);

    const auto group = warpcore::cg::tiled_partition<1>(warpcore::cg::this_thread_block());
    hashSet.insert(seed, group);

}

enum class Op { Not, And, Or, Xor, LShift, RShift, Neg, Plus, Minus, Mul, Div, Mod };

template<Op op>
__device__ void applyOperator(
    uint32_t* CS,
    uint32_t* d_MBACache,
    int ldx, int rdx,
    const int numOfSamples)
{

    if constexpr (op == Op::Not) {
        for (int i = 0; i < numOfSamples; ++i) {
            CS[i] = ~d_MBACache[ldx * numOfSamples + i];
        }
    } else if constexpr (op == Op::And) {
        for (int i = 0; i < numOfSamples; ++i) {
            CS[i] = d_MBACache[ldx * numOfSamples + i] & d_MBACache[rdx * numOfSamples + i];
        }
    } else if constexpr (op == Op::Or) {
        for (int i = 0; i < numOfSamples; ++i) {
            CS[i] = d_MBACache[ldx * numOfSamples + i] | d_MBACache[rdx * numOfSamples + i];
        }
    } else if constexpr (op == Op::Xor) {
        for (int i = 0; i < numOfSamples; ++i) {
            CS[i] = d_MBACache[ldx * numOfSamples + i] ^ d_MBACache[rdx * numOfSamples + i];
        }
    } else if constexpr (op == Op::LShift) {
        for (int i = 0; i < numOfSamples; ++i) {
            CS[i] = d_MBACache[ldx * numOfSamples + i] << d_MBACache[rdx * numOfSamples + i];
        }
    } else if constexpr (op == Op::RShift) {
        for (int i = 0; i < numOfSamples; ++i) {
            CS[i] = d_MBACache[ldx * numOfSamples + i] >> d_MBACache[rdx * numOfSamples + i];
        }
    } else if constexpr (op == Op::Neg) {
        for (int i = 0; i < numOfSamples; ++i) {
            CS[i] = -d_MBACache[ldx * numOfSamples + i];
        }
    } else if constexpr (op == Op::Plus) {
        for (int i = 0; i < numOfSamples; ++i) {
            CS[i] = d_MBACache[ldx * numOfSamples + i] + d_MBACache[rdx * numOfSamples + i];
        }
    } else if constexpr (op == Op::Minus) {
        for (int i = 0; i < numOfSamples; ++i) {
            CS[i] = d_MBACache[ldx * numOfSamples + i] - d_MBACache[rdx * numOfSamples + i];
        }
    } else if constexpr (op == Op::Mul) {
        for (int i = 0; i < numOfSamples; ++i) {
            CS[i] = d_MBACache[ldx * numOfSamples + i] * d_MBACache[rdx * numOfSamples + i];
        }
    } else if constexpr (op == Op::Div) {
        for (int i = 0; i < numOfSamples; ++i) {
            CS[i] = d_MBACache[ldx * numOfSamples + i] / d_MBACache[rdx * numOfSamples + i];
        }
    } else if constexpr (op == Op::Mod) {
        for (int i = 0; i < numOfSamples; ++i) {
            CS[i] = d_MBACache[ldx * numOfSamples + i] % d_MBACache[rdx * numOfSamples + i];
        }
    } else {
        [] <bool flag = false>() { static_assert(flag, "Unhandled operator"); }();
    }

}

template<class hash_set_t>
__device__ bool processUniqueCS(
    uint32_t* CS,
    const int numOfSamples,
    hash_set_t& hashSet)
{

    uint64_t seed{};
    makeUnqChk(CS, seed, numOfSamples);

    const auto group = warpcore::cg::tiled_partition<1>(warpcore::cg::this_thread_block());
    return (hashSet.insert(seed, group) > 0) ? false : true;

}

__device__ void insertInCache(
    bool CS_is_unique,
    uint32_t* CS,
    int tid, int ldx, int rdx,
    const int numOfSamples,
    uint32_t* d_temp_MBACache,
    int* d_temp_leftIdx, int* d_temp_rightIdx,
    int* d_FinalMBAIdx)
{

    if (CS_is_unique) {

        for (int i = 0; i < numOfSamples; ++i) {
            d_temp_MBACache[tid * numOfSamples + i] = CS[i];
        }
        d_temp_leftIdx[tid] = ldx; d_temp_rightIdx[tid] = rdx;

        bool found = true;
        for (int i = 0; found && i < numOfSamples; ++i) found = (CS[i] == d_outputData[i]);
        if (found) atomicCAS(d_FinalMBAIdx, -1, tid);

    } else {

        for (int i = 0; i < numOfSamples; ++i) {
            d_temp_MBACache[tid * numOfSamples + i] = (uint32_t)-1;
        }
        d_temp_leftIdx[tid] = -1; d_temp_rightIdx[tid] = -1;

    }

}

template<Op op, class hash_set_t>
__global__ void processOperator(
    const int idx1, const int idx2,
    const int idx3, const int idx4,
    const int numOfSamples,
    uint32_t* d_MBACache, uint32_t* d_temp_MBACache,
    int* d_temp_leftIdx, int* d_temp_rightIdx,
    int* d_FinalMBAIdx,
    hash_set_t hashSet)
{

    const int tid = (blockDim.x * blockIdx.x + threadIdx.x);
    constexpr bool isUnary = (op == Op::Not || op == Op::Neg);
    constexpr bool notCommut = (op == Op::Minus || op == Op::Div || op == Op::Mod || op == Op::LShift || op == Op::RShift);
    int maxTid = isUnary ? (idx2 - idx1 + 1) : (idx4 - idx3 + 1) * (idx2 - idx1 + 1);

    if (tid < maxTid) {

        int ldx = isUnary ? idx1 + tid : idx1 + tid / (idx4 - idx3 + 1);
        int rdx = isUnary ? 0 : idx3 + tid % (idx4 - idx3 + 1);
        const int modifiedTid = notCommut ? (tid * 2) : tid;
        uint32_t CS[maxNumOfSamples];
        applyOperator<op>(CS, d_MBACache, ldx, rdx, numOfSamples);

        bool CS_is_unique = processUniqueCS(CS, numOfSamples, hashSet);
        insertInCache(
            CS_is_unique, CS, modifiedTid, ldx, rdx, numOfSamples,
            d_temp_MBACache, d_temp_leftIdx, d_temp_rightIdx, d_FinalMBAIdx
        );

        if (notCommut) {

            applyOperator<op>(CS, d_MBACache, rdx, ldx, numOfSamples);

            bool CS_is_unique = processUniqueCS(CS, numOfSamples, hashSet);
            insertInCache(
                CS_is_unique, CS, modifiedTid + 1, rdx, ldx, numOfSamples,
                d_temp_MBACache, d_temp_leftIdx, d_temp_rightIdx, d_FinalMBAIdx
            );

        }

    }

}

// Transfering the unique CSs from temporary cache to MBACache
bool storeUniqueMBAs(
    int N,
    int& lastIdx,
    const int numOfSamples,
    const int MBACacheCapacity,
    uint32_t* d_MBACache,
    uint32_t* d_temp_MBACache,
    int* d_leftIdx, int* d_rightIdx,
    int* d_temp_leftIdx, int* d_temp_rightIdx)
{

    thrust::device_ptr<uint32_t> new_end_ptr;
    thrust::device_ptr<uint32_t> d_MBACache_ptr(d_MBACache + lastIdx * numOfSamples);
    thrust::device_ptr<uint32_t> d_temp_MBACache_ptr(d_temp_MBACache);
    thrust::device_ptr<int> d_leftIdx_ptr(d_leftIdx + lastIdx);
    thrust::device_ptr<int> d_rightIdx_ptr(d_rightIdx + lastIdx);
    thrust::device_ptr<int> d_temp_leftIdx_ptr(d_temp_leftIdx);
    thrust::device_ptr<int> d_temp_rightIdx_ptr(d_temp_rightIdx);

    new_end_ptr =
        thrust::remove(d_temp_MBACache_ptr, d_temp_MBACache_ptr + N * numOfSamples, (uint32_t)-1);
    thrust::remove(d_temp_leftIdx_ptr, d_temp_leftIdx_ptr + N, -1);
    thrust::remove(d_temp_rightIdx_ptr, d_temp_rightIdx_ptr + N, -1);

    int numberOfNewUniqueMBAs = static_cast<int>(new_end_ptr - d_temp_MBACache_ptr) / numOfSamples;
    if (lastIdx + numberOfNewUniqueMBAs > MBACacheCapacity) {
        return true;
    } else {
        N = numberOfNewUniqueMBAs;
        thrust::copy_n(d_temp_MBACache_ptr, N * numOfSamples, d_MBACache_ptr);
        thrust::copy_n(d_temp_leftIdx_ptr, N, d_leftIdx_ptr);
        thrust::copy_n(d_temp_rightIdx_ptr, N, d_rightIdx_ptr);
        lastIdx += N;
        return false;
    }

}

// Finding the left and right indices that makes the final MBA to bring to the host later
__global__ void generateResIndices(
    const int numVar,
    const int index,
    const int* d_leftIdx,
    const int* d_rightIdx,
    int* d_FinalMBAIdx)
{

    int resIdx = 0;
    while (d_FinalMBAIdx[resIdx] != -1) resIdx++;
    int queue[100];
    queue[0] = index;
    int head = 0;
    int tail = 1;
    while (head < tail) {
        int mba = queue[head];
        int l = d_leftIdx[mba];
        int r = d_rightIdx[mba];
        d_FinalMBAIdx[resIdx++] = mba;
        d_FinalMBAIdx[resIdx++] = l;
        d_FinalMBAIdx[resIdx++] = r;
        if (l >= numVar) queue[tail++] = l;
        if (r >= numVar) queue[tail++] = r;
        head++;
    }

}

// Generating the final RE string recursively
// When all the left and right indices are ready in the host
string toString(
    int index,
    map<int, pair<int, int>>& indicesMap,
    const int* startPoints)
{

    if (index < numVar) { return "v" + to_string(index); }

    int i = 0;
    while (index >= startPoints[i]) { i++; }
    i--;

    if (i % 12 == 0) {
        string res = toString(indicesMap[index].first, indicesMap, startPoints);
        return "~(" + res + ")";
    }

    if (i % 12 == 1) {
        string left = toString(indicesMap[index].first, indicesMap, startPoints);
        string right = toString(indicesMap[index].second, indicesMap, startPoints);
        return "(" + left + ")" + "&" + "(" + right + ")";
    }

    if (i % 12 == 2) {
        string left = toString(indicesMap[index].first, indicesMap, startPoints);
        string right = toString(indicesMap[index].second, indicesMap, startPoints);
        return "(" + left + ")" + "|" + "(" + right + ")";
    }

    if (i % 12 == 3) {
        string left = toString(indicesMap[index].first, indicesMap, startPoints);
        string right = toString(indicesMap[index].second, indicesMap, startPoints);
        return "(" + left + ")" + "^" + "(" + right + ")";
    }

    if (i % 12 == 4) {
        string left = toString(indicesMap[index].first, indicesMap, startPoints);
        string right = toString(indicesMap[index].second, indicesMap, startPoints);
        return "(" + left + ")" + "<<" + "(" + right + ")";
    }

    if (i % 12 == 5) {
        string left = toString(indicesMap[index].first, indicesMap, startPoints);
        string right = toString(indicesMap[index].second, indicesMap, startPoints);
        return "(" + left + ")" + ">>" + "(" + right + ")";
    }

    if (i % 12 == 6) {
        string res = toString(indicesMap[index].first, indicesMap, startPoints);
        return "-(" + res + ")";
    }

    if (i % 12 == 7) {
        string left = toString(indicesMap[index].first, indicesMap, startPoints);
        string right = toString(indicesMap[index].second, indicesMap, startPoints);
        return "(" + left + ")" + "+" + "(" + right + ")";
    }

    if (i % 12 == 8) {
        string left = toString(indicesMap[index].first, indicesMap, startPoints);
        string right = toString(indicesMap[index].second, indicesMap, startPoints);
        return "(" + left + ")" + "-" + "(" + right + ")";
    }

    if (i % 12 == 9) {
        string left = toString(indicesMap[index].first, indicesMap, startPoints);
        string right = toString(indicesMap[index].second, indicesMap, startPoints);
        return "(" + left + ")" + "*" + "(" + right + ")";
    }

    if (i % 12 == 10) {
        string left = toString(indicesMap[index].first, indicesMap, startPoints);
        string right = toString(indicesMap[index].second, indicesMap, startPoints);
        return "(" + left + ")" + "/" + "(" + right + ")";
    }

    else {
        string left = toString(indicesMap[index].first, indicesMap, startPoints);
        string right = toString(indicesMap[index].second, indicesMap, startPoints);
        return "(" + left + ")" + "%" + "(" + right + ")";
    }

}

// Bringing the left and right indices of the MBA from device to host
string MBAToString(
    const int FinalMBAIdx,
    const int lastIdx,
    const int* startPoints,
    const int* d_leftIdx, const int* d_rightIdx,
    const int* d_temp_leftIdx, const int* d_temp_rightIdx)
{

    auto* LIdx = new int[1];
    auto* RIdx = new int[1];

    checkCuda(cudaMemcpy(LIdx, d_temp_leftIdx + FinalMBAIdx, sizeof(int), cudaMemcpyDeviceToHost));
    checkCuda(cudaMemcpy(RIdx, d_temp_rightIdx + FinalMBAIdx, sizeof(int), cudaMemcpyDeviceToHost));

    int* d_resIndices;
    checkCuda(cudaMalloc(&d_resIndices, 100 * sizeof(int)));

    thrust::device_ptr<int> d_resIndices_ptr(d_resIndices);
    thrust::fill(d_resIndices_ptr, d_resIndices_ptr + 100, -1);

    if (*LIdx >= numVar) generateResIndices << <1, 1 >> > (numVar, *LIdx, d_leftIdx, d_rightIdx, d_resIndices);
    if (*RIdx >= numVar) generateResIndices << <1, 1 >> > (numVar, *RIdx, d_leftIdx, d_rightIdx, d_resIndices);

    int resIndices[100];
    checkCuda(cudaMemcpy(resIndices, d_resIndices, 100 * sizeof(int), cudaMemcpyDeviceToHost));

    map<int, pair<int, int>> indicesMap;
    indicesMap.insert(make_pair(INT_MAX - 1, make_pair(*LIdx, *RIdx)));

    int i = 0;
    while (resIndices[i] != -1 && i + 2 < 100) {
        int mba = resIndices[i];
        int l = resIndices[i + 1];
        int r = resIndices[i + 2];
        indicesMap.insert(make_pair(mba, make_pair(l, r)));
        i += 3;
    }

    cudaFree(d_resIndices);

    if (i + 2 >= 100) return "Size of the output is too big !";
    else return toString(INT_MAX - 1, indicesMap, startPoints);

}

template<Op op, class hash_set_t>
bool generateMBAs(
    int MBALen,
    int* startPoints,
    const int temp_MBACacheCapacity, const int MBACacheCapacity,
    uint64_t& allMBAs,
    int& lastIdx,
    const int numOfSamples,
    uint32_t* d_MBACache, uint32_t* d_temp_MBACache,
    int* d_temp_leftIdx, int* d_temp_rightIdx,
    int* d_leftIdx, int* d_rightIdx,
    int* d_FinalMBAIdx, int* FinalMBAIdx,
    hash_set_t hashSet,
    const char* opStr, int opIdx)
{

    constexpr bool isUnary = (op == Op::Not || op == Op::Neg);
    constexpr bool notCommut = (op == Op::Minus || op == Op::Div || op == Op::Mod || op == Op::LShift || op == Op::RShift);

    bool lastRound = false;

    if (isUnary) {

        int idx1 = startPoints[(MBALen - 1) * 12];
        int idx2 = startPoints[MBALen * 12] - 1;
        int N = idx2 - idx1 + 1;

        if (N) {
            int x = idx1, y;
            do {
                y = x + min(temp_MBACacheCapacity - 1, idx2 - x);
                N = y - x + 1;
#ifndef MEASUREMENT_MODE
                printf("Length %-2d | %s | AllMBAs: %-11lu | StoredMBAs: %-10d | ToBeChecked: %-10d \n",
                    MBALen, opStr, allMBAs, lastIdx, N);
#endif
                int Blc = (N + 1023) / 1024;
                processOperator<op, hash_set_t> << <Blc, 1024 >> > (
                    x, y, 0, 0, numOfSamples, d_MBACache, d_temp_MBACache,
                    d_temp_leftIdx, d_temp_rightIdx, d_FinalMBAIdx, hashSet
                    );
                checkCuda(cudaPeekAtLastError());
                checkCuda(cudaMemcpy(FinalMBAIdx, d_FinalMBAIdx, sizeof(int), cudaMemcpyDeviceToHost));
                allMBAs += N;
                if (*FinalMBAIdx != -1) { startPoints[MBALen * 12 + opIdx] = INT_MAX; return true; }
                lastRound = storeUniqueMBAs(
                    N, lastIdx, numOfSamples, MBACacheCapacity, d_MBACache, d_temp_MBACache,
                    d_leftIdx, d_rightIdx, d_temp_leftIdx, d_temp_rightIdx
                );
                x = y + 1;
            } while (y < idx2 && !(lastRound));
        }
        startPoints[MBALen * 12 + opIdx] = lastIdx;

    } else {

        for (int i = 1; 2 * i <= MBALen - 1; ++i) {

            int idx1 = startPoints[i * 12];
            int idx2 = startPoints[(i + 1) * 12] - 1;
            int idx3 = startPoints[(MBALen - i - 1) * 12];
            int idx4 = startPoints[(MBALen - i) * 12] - 1;
            int N = (idx4 - idx3 + 1) * (idx2 - idx1 + 1);
            int modCap = notCommut ? (idx2 - idx1 + 1) - 1 : 2 * (idx2 - idx1 + 1) - 1;

            if (N) {
                int x = idx3, y;
                do {
                    y = x + min(temp_MBACacheCapacity / modCap, idx4 - x);
                    N = (y - x + 1) * (idx2 - idx1 + 1);
                    int modN = notCommut ? 2 * N : N;
#ifndef MEASUREMENT_MODE
                    printf("Length %-2d | %s | AllMBAs: %-11lu | StoredMBAs: %-10d | ToBeChecked: %-10d \n",
                        MBALen, opStr, allMBAs, lastIdx, modN);
#endif
                    int Blc = (N + 1023) / 1024;
                    processOperator<op, hash_set_t> << <Blc, 1024 >> > (
                        idx1, idx2, x, y, numOfSamples, d_MBACache, d_temp_MBACache,
                        d_temp_leftIdx, d_temp_rightIdx, d_FinalMBAIdx, hashSet
                        );
                    checkCuda(cudaPeekAtLastError());
                    checkCuda(cudaMemcpy(FinalMBAIdx, d_FinalMBAIdx, sizeof(int), cudaMemcpyDeviceToHost));
                    allMBAs += modN;
                    if (*FinalMBAIdx != -1) { startPoints[MBALen * 12 + opIdx] = INT_MAX; return true; }
                    lastRound = storeUniqueMBAs(
                        modN, lastIdx, numOfSamples, MBACacheCapacity, d_MBACache, d_temp_MBACache,
                        d_leftIdx, d_rightIdx, d_temp_leftIdx, d_temp_rightIdx
                    );
                    x = y + 1;
                } while (y < idx4 && !(lastRound));
            }

        }

        startPoints[MBALen * 12 + opIdx] = lastIdx;

    }

    return lastRound;

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

    // Number of generated formulas
    uint64_t allMBAs{};

    // Index of the last free position in the cache
    int lastIdx{};

#ifndef MEASUREMENT_MODE
    printf("Length %-2d | (V)   | AllMBAs: %-11lu | StoredMBAs: %-10d | ToBeChecked: %-10d \n",
        1, allMBAs, 0, numVar);
#endif

    // Initializing the cache with variables
    int index{};
    for (int i = 0; i < numVar; ++i) {
        bool found = true;
        for (int j = 0; j < numOfSamples; j++) {
            MBACache[index++] = inputData[j * numVar + i];
            if (!(inputData[j * numVar + i] = outputData[j])) found = false;
        }
        allMBAs++; lastIdx++;
        if (found) return "v" + to_string(i);
    }

    // ------------------------------------
    // Memory allocation and initialisation
    // ------------------------------------

    int maxAllocationSize;
    cudaDeviceGetAttribute(&maxAllocationSize, cudaDevAttrMaxPitch, 0);

    const int MBACacheCapacity = maxAllocationSize / (numOfSamples * sizeof(uint32_t)) * 1.5;
    const int temp_MBACacheCapacity = MBACacheCapacity / 2;

    // Unary operators : ~, Neg
    // Binary operators : &, |, ^, <<, >>, +, -, *, /, %
    int* startPoints = new int[(maxLen + 2) * 12]();
    startPoints[(2 * 12) - 1] = lastIdx;
    startPoints[2 * 12] = lastIdx;

    int* d_FinalMBAIdx;
    int* FinalMBAIdx = new int[1]; *FinalMBAIdx = -1;
    checkCuda(cudaMalloc(&d_FinalMBAIdx, sizeof(int)));
    checkCuda(cudaMemcpy(d_FinalMBAIdx, FinalMBAIdx, sizeof(int), cudaMemcpyHostToDevice));

    uint32_t* d_MBACache, * d_temp_MBACache;
    int* d_leftIdx, * d_rightIdx, * d_temp_leftIdx, * d_temp_rightIdx;
    checkCuda(cudaMalloc(&d_leftIdx, MBACacheCapacity * sizeof(int)));
    checkCuda(cudaMalloc(&d_rightIdx, MBACacheCapacity * sizeof(int)));
    checkCuda(cudaMalloc(&d_temp_leftIdx, temp_MBACacheCapacity * sizeof(int)));
    checkCuda(cudaMalloc(&d_temp_rightIdx, temp_MBACacheCapacity * sizeof(int)));
    checkCuda(cudaMalloc(&d_MBACache, MBACacheCapacity * numOfSamples * sizeof(uint32_t)));
    checkCuda(cudaMalloc(&d_temp_MBACache, temp_MBACacheCapacity * numOfSamples * sizeof(uint32_t)));

    using hash_set_t = warpcore::HashSet<
        uint64_t,         // Key type
        uint64_t(0) - 1,  // Empty key
        uint64_t(0) - 2,  // Tombstone key
        warpcore::probing_schemes::QuadraticProbing<warpcore::hashers::MurmurHash <uint64_t>>>;

    hash_set_t hashSet(MBACacheCapacity);

    checkCuda(cudaMemcpy(d_MBACache, MBACache, numVar * numOfSamples * sizeof(uint32_t), cudaMemcpyHostToDevice));
    hashSetInit<hash_set_t> << <1, numVar >> > (numOfSamples, hashSet, d_MBACache);

    // ----------------------------
    // Enumeration of the next MBAs
    // ----------------------------

    bool lastRound = false;

    for (int MBALen = 2; MBALen <= maxLen; ++MBALen) {

        if (MBALen == 3) {
            printDeviceArray(d_MBACache, lastIdx * numOfSamples, numOfSamples);
        }

        lastRound = generateMBAs<Op::Not>(MBALen, startPoints, temp_MBACacheCapacity, MBACacheCapacity, allMBAs, lastIdx,
            numOfSamples, d_MBACache, d_temp_MBACache, d_temp_leftIdx, d_temp_rightIdx,
            d_leftIdx, d_rightIdx, d_FinalMBAIdx, FinalMBAIdx, hashSet, "(~)  ", 1);
        if (lastRound) break;

        lastRound = generateMBAs<Op::And>(MBALen, startPoints, temp_MBACacheCapacity, MBACacheCapacity, allMBAs, lastIdx,
            numOfSamples, d_MBACache, d_temp_MBACache, d_temp_leftIdx, d_temp_rightIdx,
            d_leftIdx, d_rightIdx, d_FinalMBAIdx, FinalMBAIdx, hashSet, "(&)  ", 2);
        if (lastRound) break;

        lastRound = generateMBAs<Op::Or>(MBALen, startPoints, temp_MBACacheCapacity, MBACacheCapacity, allMBAs, lastIdx,
            numOfSamples, d_MBACache, d_temp_MBACache, d_temp_leftIdx, d_temp_rightIdx,
            d_leftIdx, d_rightIdx, d_FinalMBAIdx, FinalMBAIdx, hashSet, "(|)  ", 3);
        if (lastRound) break;

        lastRound = generateMBAs<Op::Xor>(MBALen, startPoints, temp_MBACacheCapacity, MBACacheCapacity, allMBAs, lastIdx,
            numOfSamples, d_MBACache, d_temp_MBACache, d_temp_leftIdx, d_temp_rightIdx,
            d_leftIdx, d_rightIdx, d_FinalMBAIdx, FinalMBAIdx, hashSet, "(^)  ", 4);
        if (lastRound) break;

        lastRound = generateMBAs<Op::LShift>(MBALen, startPoints, temp_MBACacheCapacity, MBACacheCapacity, allMBAs, lastIdx,
            numOfSamples, d_MBACache, d_temp_MBACache, d_temp_leftIdx, d_temp_rightIdx,
            d_leftIdx, d_rightIdx, d_FinalMBAIdx, FinalMBAIdx, hashSet, "(<<) ", 5);
        if (lastRound) break;

        lastRound = generateMBAs<Op::RShift>(MBALen, startPoints, temp_MBACacheCapacity, MBACacheCapacity, allMBAs, lastIdx,
            numOfSamples, d_MBACache, d_temp_MBACache, d_temp_leftIdx, d_temp_rightIdx,
            d_leftIdx, d_rightIdx, d_FinalMBAIdx, FinalMBAIdx, hashSet, "(>>) ", 6);
        if (lastRound) break;

        lastRound = generateMBAs<Op::Neg>(MBALen, startPoints, temp_MBACacheCapacity, MBACacheCapacity, allMBAs, lastIdx,
            numOfSamples, d_MBACache, d_temp_MBACache, d_temp_leftIdx, d_temp_rightIdx,
            d_leftIdx, d_rightIdx, d_FinalMBAIdx, FinalMBAIdx, hashSet, "(Neg)", 7);
        if (lastRound) break;

        lastRound = generateMBAs<Op::Plus>(MBALen, startPoints, temp_MBACacheCapacity, MBACacheCapacity, allMBAs, lastIdx,
            numOfSamples, d_MBACache, d_temp_MBACache, d_temp_leftIdx, d_temp_rightIdx,
            d_leftIdx, d_rightIdx, d_FinalMBAIdx, FinalMBAIdx, hashSet, "(+)  ", 8);
        if (lastRound) break;

        lastRound = generateMBAs<Op::Minus>(MBALen, startPoints, temp_MBACacheCapacity, MBACacheCapacity, allMBAs, lastIdx,
            numOfSamples, d_MBACache, d_temp_MBACache, d_temp_leftIdx, d_temp_rightIdx,
            d_leftIdx, d_rightIdx, d_FinalMBAIdx, FinalMBAIdx, hashSet, "(-)  ", 9);
        if (lastRound) break;

        lastRound = generateMBAs<Op::Mul>(MBALen, startPoints, temp_MBACacheCapacity, MBACacheCapacity, allMBAs, lastIdx,
            numOfSamples, d_MBACache, d_temp_MBACache, d_temp_leftIdx, d_temp_rightIdx,
            d_leftIdx, d_rightIdx, d_FinalMBAIdx, FinalMBAIdx, hashSet, "(*)  ", 10);
        if (lastRound) break;

        lastRound = generateMBAs<Op::Div>(MBALen, startPoints, temp_MBACacheCapacity, MBACacheCapacity, allMBAs, lastIdx,
            numOfSamples, d_MBACache, d_temp_MBACache, d_temp_leftIdx, d_temp_rightIdx,
            d_leftIdx, d_rightIdx, d_FinalMBAIdx, FinalMBAIdx, hashSet, "(/)  ", 11);
        if (lastRound) break;

        lastRound = generateMBAs<Op::Mod>(MBALen, startPoints, temp_MBACacheCapacity, MBACacheCapacity, allMBAs, lastIdx,
            numOfSamples, d_MBACache, d_temp_MBACache, d_temp_leftIdx, d_temp_rightIdx,
            d_leftIdx, d_rightIdx, d_FinalMBAIdx, FinalMBAIdx, hashSet, "(%)  ", 12);
        if (lastRound) break;

    }

    string output;

    if (*FinalMBAIdx != -1) {
        output = MBAToString(*FinalMBAIdx, lastIdx, startPoints,
            d_leftIdx, d_rightIdx, d_temp_leftIdx, d_temp_rightIdx);
    } else {
        output = "Not found !";
    }

    // Cleanup
    cudaFree(d_MBACache);
    cudaFree(d_temp_MBACache);
    cudaFree(d_leftIdx);
    cudaFree(d_rightIdx);
    cudaFree(d_temp_leftIdx);
    cudaFree(d_temp_rightIdx);
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

    if (atoi(argv[2]) < 1 || atoi(argv[11]) > 100) {
        printf("Argument maxLen = = \"%s\" should be between 1 and 100", argv[2]);
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
    printf("\nNumber of All LTLs: %lu", allLTLs);
    printf("\nCost of Final LTL: %d", LTLcost);
    printf("\nRunning Time: %f s", (double)duration * 0.000001);
#endif
    printf("\nMBA: \"%s\"\n", output.c_str());

    return 0;
}