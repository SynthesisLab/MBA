#include <iostream>
#include <fstream>
#include <stack>
#include <thrust/copy.h>
#include <thrust/remove.h>
#include <thrust/device_ptr.h>
#include <warpcore/hash_set.cuh>
#include "json.hpp"

// #define MEASUREMENT_MODE

using namespace std;
using json = nlohmann::json;

const int maxNumOfVars = 10;
const int maxNumOfSamples = 100;
int numVar;

__constant__ int d_numVar;
__constant__ uint32_t d_inputData[maxNumOfVars * maxNumOfSamples];
__constant__ uint32_t d_outputData[maxNumOfSamples];
__device__ __constant__ char d_opChar[12] = { '~', '&', '|', '^', '<', '>', '#', '+', '-', '*', '/', '%' };

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
    hash_set_t hashSet)
{

    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t CS[maxNumOfSamples];

    for (int i = 0; i < numOfSamples; ++i) {
        CS[i] = d_inputData[i * d_numVar + tid];
    }

    uint64_t seed{};
    makeUnqChk(CS, seed, numOfSamples);

    const auto group = warpcore::cg::tiled_partition<1>(warpcore::cg::this_thread_block());
    hashSet.insert(seed, group);

}

enum class Op { Not, And, Or, Xor, LShift, RShift, Neg, Plus, Minus, Mul, Div, Mod };

__device__ uint32_t evaluateRPN(char* d_MBACache, int idx, int sampleIdx)
{

    uint32_t stack[32];
    int stackIdx = -1;

    while (true) {

        char token = d_MBACache[idx++];

        if (token == '\0') return stack[0];
        else if (token >= '0' && token <= '9') {
            int var = token - '0';
            stack[++stackIdx] = d_inputData[sampleIdx * d_numVar + var];
        } else {
            uint32_t left, right;
            if (token == '~' || token == '#') {
                left = stack[stackIdx];
                if (token == '~') stack[stackIdx] = ~left;
                else stack[stackIdx] = -left;
            } else {
                right = stack[stackIdx--];
                left = stack[stackIdx];
                switch (token) {
                case '&': stack[stackIdx] = left & right; break;
                case '|': stack[stackIdx] = left | right; break;
                case '^': stack[stackIdx] = left ^ right; break;
                case '<': stack[stackIdx] = left << right; break;
                case '>': stack[stackIdx] = left >> right; break;
                case '+': stack[stackIdx] = left + right; break;
                case '-': stack[stackIdx] = left - right; break;
                case '*': stack[stackIdx] = left * right; break;
                case '/': stack[stackIdx] = left / right; break;
                case '%': stack[stackIdx] = left % right; break;
                }
            }
        }

    }

}

template<Op op>
__device__ void evaluateFormula(
    uint32_t* CS,
    char* d_MBACache, int ldx, int rdx,
    const int numOfSamples)
{

    uint32_t left, right;

    if constexpr (op == Op::Not) {
        for (int i = 0; i < numOfSamples; ++i) {
            left = evaluateRPN(d_MBACache, ldx, i);
            CS[i] = ~left;
        }
    } else if constexpr (op == Op::And) {
        for (int i = 0; i < numOfSamples; ++i) {
            left = evaluateRPN(d_MBACache, ldx, i);
            right = evaluateRPN(d_MBACache, rdx, i);
            CS[i] = left & right;
        }
    } else if constexpr (op == Op::Or) {
        for (int i = 0; i < numOfSamples; ++i) {
            left = evaluateRPN(d_MBACache, ldx, i);
            right = evaluateRPN(d_MBACache, rdx, i);
            CS[i] = left | right;
        }
    } else if constexpr (op == Op::Xor) {
        for (int i = 0; i < numOfSamples; ++i) {
            left = evaluateRPN(d_MBACache, ldx, i);
            right = evaluateRPN(d_MBACache, rdx, i);
            CS[i] = left ^ right;
        }
    } else if constexpr (op == Op::LShift) {
        for (int i = 0; i < numOfSamples; ++i) {
            left = evaluateRPN(d_MBACache, ldx, i);
            right = evaluateRPN(d_MBACache, rdx, i);
            CS[i] = left << right;
        }
    } else if constexpr (op == Op::RShift) {
        for (int i = 0; i < numOfSamples; ++i) {
            left = evaluateRPN(d_MBACache, ldx, i);
            right = evaluateRPN(d_MBACache, rdx, i);
            CS[i] = left >> right;
        }
    } else if constexpr (op == Op::Neg) {
        for (int i = 0; i < numOfSamples; ++i) {
            left = evaluateRPN(d_MBACache, ldx, i);
            CS[i] = -left;
        }
    } else if constexpr (op == Op::Plus) {
        for (int i = 0; i < numOfSamples; ++i) {
            left = evaluateRPN(d_MBACache, ldx, i);
            right = evaluateRPN(d_MBACache, rdx, i);
            CS[i] = left + right;
        }
    } else if constexpr (op == Op::Minus) {
        for (int i = 0; i < numOfSamples; ++i) {
            left = evaluateRPN(d_MBACache, ldx, i);
            right = evaluateRPN(d_MBACache, rdx, i);
            CS[i] = left - right;
        }
    } else if constexpr (op == Op::Mul) {
        for (int i = 0; i < numOfSamples; ++i) {
            left = evaluateRPN(d_MBACache, ldx, i);
            right = evaluateRPN(d_MBACache, rdx, i);
            CS[i] = left * right;
        }
    } else if constexpr (op == Op::Div) {
        bool noZeros = true;
        for (int i = 0; i < numOfSamples; ++i) {
            left = evaluateRPN(d_MBACache, ldx, i);
            right = evaluateRPN(d_MBACache, rdx, i);
            if (right == 0) noZeros = false;
            else CS[i] = left / right;
        }
        if (!noZeros) {
            for (int i = 0; i < numOfSamples; ++i) {
                CS[i] = evaluateRPN(d_MBACache, ldx, i);
            }
        }
    } else if constexpr (op == Op::Mod) {
        bool noZeros = true;
        for (int i = 0; i < numOfSamples; ++i) {
            left = evaluateRPN(d_MBACache, ldx, i);
            right = evaluateRPN(d_MBACache, rdx, i);
            if (right == 0) noZeros = false;
            else CS[i] = left % right;
        }
        if (!noZeros) {
            for (int i = 0; i < numOfSamples; ++i) {
                CS[i] = evaluateRPN(d_MBACache, ldx, i);
            }
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

template<Op op>
__device__ void insertInCache(
    uint32_t* CS, bool CSUniq,
    int tid, int ldx, int rdx,
    int llen, int rlen, int len,
    const int numOfSamples,
    char* d_MBACache, char* d_temp_MBACache,
    char* d_MBAFormula)
{

    if (CSUniq) {

        for (int i = 0; i < llen - 1; ++i) d_temp_MBACache[tid * len + i] = d_MBACache[ldx + i];
        for (int i = 0; i < rlen - 1; ++i) d_temp_MBACache[tid * len + i + llen - 1] = d_MBACache[rdx + i];
        d_temp_MBACache[tid * len + len - 2] = d_opChar[static_cast<int>(op)];
        d_temp_MBACache[tid * len + len - 1] = '\0';

        bool found = true;
        for (int i = 0; found && i < numOfSamples; ++i) found = (CS[i] == d_outputData[i]);
        if (found) {
            for (int i = 0; i < len; ++i) d_MBAFormula[i] = d_temp_MBACache[tid * len + i];
        }

    } else {

        for (int i = 0; i < len; ++i) d_temp_MBACache[tid * len + i] = '?';

    }

}

template<Op op, class hash_set_t>
__global__ void processOperator(
    char* d_MBACache, char* d_temp_MBACache,
    const int ldx1, const int ldx2, const int llen,
    const int rdx1, const int rdx2, const int rlen,
    const int numOfSamples,
    char* d_MBAFormula,
    hash_set_t hashSet)
{

    const int lnumOfFormulas = (ldx2 - ldx1 + 1) / llen;
    const int rnumOfFormulas = (rdx2 - rdx1 + 1) / rlen;
    const int tid = (blockDim.x * blockIdx.x + threadIdx.x);
    constexpr bool isUnary = (op == Op::Not || op == Op::Neg);
    constexpr bool notCommut = (op == Op::Minus || op == Op::Div || op == Op::Mod || op == Op::LShift || op == Op::RShift);
    int maxTid = isUnary ? lnumOfFormulas : lnumOfFormulas * rnumOfFormulas;

    if (tid < maxTid) {

        int ldx = isUnary ? ldx1 + tid * llen : ldx1 + tid / rnumOfFormulas * llen;
        int rdx = isUnary ? 0 : rdx1 + tid % rnumOfFormulas * rlen;
        int modTid = notCommut ? 2 * tid : tid;
        int len = llen + rlen;

        uint32_t CS[maxNumOfSamples];
        evaluateFormula<op>(CS, d_MBACache, ldx, rdx, numOfSamples);
        bool CSUniq = processUniqueCS(CS, numOfSamples, hashSet);
        insertInCache<op>(CS, CSUniq, modTid, ldx, rdx, llen, rlen, len, numOfSamples, d_MBACache, d_temp_MBACache, d_MBAFormula);

        if (notCommut) {

            modTid += 1;
            evaluateFormula<op>(CS, d_MBACache, rdx, ldx, numOfSamples);
            CSUniq = processUniqueCS(CS, numOfSamples, hashSet);
            insertInCache<op>(CS, CSUniq, modTid, rdx, ldx, rlen, llen, len, numOfSamples, d_MBACache, d_temp_MBACache, d_MBAFormula);

        }

    }

}

bool storeUniqueMBAs(
    int N,
    uint64_t& lastIdx,
    int MBALen,
    const int numOfSamples,
    const int MBACacheCapacity,
    char* d_MBACache, char* d_temp_MBACache)
{

    int space = N * (MBALen + 1);

    thrust::device_ptr<char> new_end_ptr;
    thrust::device_ptr<char> d_MBACache_ptr(d_MBACache + lastIdx);
    thrust::device_ptr<char> d_temp_MBACache_ptr(d_temp_MBACache);

    new_end_ptr = thrust::remove(d_temp_MBACache_ptr, d_temp_MBACache_ptr + space, '?');
    space = static_cast<int>(new_end_ptr - d_temp_MBACache_ptr);

    if (lastIdx + space > MBACacheCapacity) {
        return true;
    } else {
        thrust::copy_n(d_temp_MBACache_ptr, space, d_MBACache_ptr);
        lastIdx += space;
        return false;
    }

}

template<Op op, class hash_set_t>
bool generateMBAs(
    int MBALen,
    int* startPoints,
    const int MBACacheCapacity, const int temp_MBACacheCapacity,
    uint64_t& allMBAs,
    uint64_t& lastIdx,
    const int numOfSamples,
    char* d_MBACache, char* d_temp_MBACache,
    char* d_MBAFormula, char* MBAFormula,
    const char* opStr,
    hash_set_t hashSet)
{

    constexpr bool isUnary = (op == Op::Not || op == Op::Neg);
    constexpr bool notCommut = (op == Op::Minus || op == Op::Div || op == Op::Mod || op == Op::LShift || op == Op::RShift);

    if (isUnary) {

        int idx1 = startPoints[MBALen - 1];
        int idx2 = startPoints[MBALen] - 1;
        int N = (idx2 - idx1 + 1) / MBALen;
        int lim = temp_MBACacheCapacity / (MBALen + 1);

        if (N) {
            int x = idx1, y;
            do {
                y = x + min(lim * MBALen - 1, idx2 - x);
                N = (y - x + 1) / MBALen;
#ifndef MEASUREMENT_MODE
                printf("Length %-2d | %s | StoredMBAs: %-11lu | ToBeChecked: %-10d \n", MBALen, opStr, allMBAs, N);
#endif
                int Blc = (N + 1023) / 1024;
                processOperator<op> << <Blc, 1024 >> > (d_MBACache, d_temp_MBACache, x, y, MBALen, 0, 0, 1, numOfSamples, d_MBAFormula, hashSet);
                checkCuda(cudaPeekAtLastError());
                checkCuda(cudaMemcpy(MBAFormula, d_MBAFormula, 64 * sizeof(char), cudaMemcpyDeviceToHost));
                if (MBAFormula[0] != '\0') return true;
                uint64_t oldIdx = lastIdx;
                if (storeUniqueMBAs(N, lastIdx, MBALen, numOfSamples, MBACacheCapacity, d_MBACache, d_temp_MBACache))
                    return true;
                allMBAs += (lastIdx - oldIdx) / (MBALen + 1);
                x = y + 1;
            } while (y < idx2);
        }

    } else {

        for (int i = 1; 2 * i <= MBALen - 1; ++i) {

            int idx1 = startPoints[i];
            int idx2 = startPoints[i + 1] - 1;
            int idx3 = startPoints[MBALen - i - 1];
            int idx4 = startPoints[MBALen - i] - 1;
            int N = ((idx4 - idx3 + 1) / (MBALen - i)) * ((idx2 - idx1 + 1) / (i + 1));
            int lim = temp_MBACacheCapacity / (((idx2 - idx1 + 1) / (i + 1)) * (MBALen + 1));
            if (notCommut) lim /= 2;

            if (N) {
                int x = idx3, y;
                do {
                    y = x + min(lim * (MBALen - i) - 1, idx4 - x);
                    N = (y - x + 1) * (idx2 - idx1 + 1) / ((i + 1) * (MBALen - i));
                    int modN = notCommut ? 2 * N : N;
#ifndef MEASUREMENT_MODE
                    printf("Length %-2d | %s | StoredMBAs: %-11lu | ToBeChecked: %-10d \n", MBALen, opStr, allMBAs, modN);
#endif
                    int Blc = (N + 1023) / 1024;
                    processOperator<op> << <Blc, 1024 >> > (d_MBACache, d_temp_MBACache, idx1, idx2, i + 1, x, y, MBALen - i, numOfSamples, d_MBAFormula, hashSet);
                    checkCuda(cudaPeekAtLastError());
                    checkCuda(cudaMemcpy(MBAFormula, d_MBAFormula, 64 * sizeof(char), cudaMemcpyDeviceToHost));
                    if (MBAFormula[0] != '\0') return true;
                    uint64_t oldIdx = lastIdx;
                    if (storeUniqueMBAs(modN, lastIdx, MBALen, numOfSamples, MBACacheCapacity, d_MBACache, d_temp_MBACache))
                        return true;
                    allMBAs += (lastIdx - oldIdx) / (MBALen + 1);
                    x = y + 1;
                } while (y < idx4);
            }

        }

    }

    startPoints[MBALen + 1] = lastIdx;
    return false;

}

string MBAToString(const char* MBAFormula) {

    stack<string> stack;
    unordered_map<char, string> opToStr = {
        {'~', "~"}, {'&', "&"}, {'|', "|"}, {'^', "^"},
        {'<', "<<"}, {'>', ">>"}, {'#', "-"}, {'+', "+"},
        {'-', "-"}, {'*', "*"}, {'/', "/"}, {'%', "%"}
    };

    for (int i = 0; MBAFormula[i] != '\0'; ++i) {

        char token = MBAFormula[i];

        if (token >= '0' && token <= '9') {
            stack.push("v" + string(1, token));
        } else {
            if (token == '~' || token == '#') {
                if (!stack.empty()) {
                    string operand = stack.top();
                    stack.pop();
                    stack.push("(" + opToStr[token] + operand + ")");
                }
            } else {
                if (stack.size() < 2) return "Error : Incorrect formula.\n";
                string right = stack.top(); stack.pop();
                string left = stack.top(); stack.pop();
                stack.push("(" + left + " " + opToStr[token] + " " + right + ")");
            }
        }

    }

    if (stack.size() != 1) return "Error : Incorrect Formula.\n";
    return stack.top();
}

void printCharsFromGPU(char* d_array, int end_index) {
    char* h_copy = new char[end_index + 1];
    cudaMemcpy(h_copy, d_array, (end_index + 1) * sizeof(char), cudaMemcpyDeviceToHost);
    cout << "Output: " << endl;
    for (int i = 0; i <= end_index; i++) {
        if (h_copy[i] == '\0') cout << endl;
        else cout << h_copy[i];
    }
    delete[] h_copy;
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

    if (numVar > 10) {
        printf("This version supports at most %d variables.\n", maxNumOfVars);
        return "see_the_error";
    }

    // Copying number of vars, inputs and outputs into the constant memory
    checkCuda(cudaMemcpyToSymbol(d_numVar, &numVar, sizeof(int)));
    checkCuda(cudaMemcpyToSymbol(d_inputData, inputData, numVar * numOfSamples * sizeof(uint32_t)));
    checkCuda(cudaMemcpyToSymbol(d_outputData, outputData, numOfSamples * sizeof(uint32_t)));

    // Creating the cache
    char* MBACache = new char[numVar * 2];

    // Number of generated formulas
    uint64_t allMBAs{};

    // Index of the last free position in the cache
    uint64_t lastIdx{};

#ifndef MEASUREMENT_MODE
    printf("Length %-2d | Vars  | StoredMBAs: %-11lu | ToBeChecked: %-10d \n", 1, allMBAs, numVar);
#endif

    // Initializing the cache with variables
    for (int i = 0; i < numVar; ++i) {
        MBACache[lastIdx++] = '0' + i;
        MBACache[lastIdx++] = '\0';
        bool found = true;
        for (int j = 0; j < numOfSamples; j++) {
            if (!(inputData[j * numVar + i] == outputData[j])) found = false;
        }
        allMBAs++;
        if (found) return "v" + to_string(i);
    }

    // ------------------------------------
    // Memory allocation and initialisation
    // ------------------------------------

    int maxAllocationSize;
    cudaDeviceGetAttribute(&maxAllocationSize, cudaDevAttrMaxPitch, 0);

    const int MBACacheCapacity = maxAllocationSize / sizeof(char);
    const int temp_MBACacheCapacity = MBACacheCapacity / 2;

    int* startPoints = new int[maxLen + 2]();
    startPoints[1] = 0;
    startPoints[2] = lastIdx;

    char* d_MBAFormula;
    char* MBAFormula = new char[64]; MBAFormula[0] = '\0';
    checkCuda(cudaMalloc(&d_MBAFormula, 64 * sizeof(char)));

    char* d_MBACache, * d_temp_MBACache;
    checkCuda(cudaMalloc(&d_MBACache, MBACacheCapacity * sizeof(char)));
    checkCuda(cudaMalloc(&d_temp_MBACache, temp_MBACacheCapacity * sizeof(char)));
    checkCuda(cudaMemcpy(d_MBACache, MBACache, 2 * numVar * sizeof(char), cudaMemcpyHostToDevice));

    using hash_set_t = warpcore::HashSet<
        uint64_t,         // Key type
        uint64_t(0) - 1,  // Empty key
        uint64_t(0) - 2,  // Tombstone key
        warpcore::probing_schemes::QuadraticProbing<warpcore::hashers::MurmurHash <uint64_t>>>;

    hash_set_t hashSet(MBACacheCapacity / 5);
    hashSet.init();
    hashSetInit << <1, numVar >> > (numOfSamples, hashSet);

    // ----------------------------
    // Enumeration of the next MBAs
    // ----------------------------

    bool lastRound = false;

    for (int MBALen = 2; MBALen <= maxLen; ++MBALen) {

        lastRound = generateMBAs<Op::Not>(MBALen, startPoints, MBACacheCapacity, temp_MBACacheCapacity, allMBAs,
            lastIdx, numOfSamples, d_MBACache, d_temp_MBACache, d_MBAFormula, MBAFormula, "(~)  ", hashSet);
        if (lastRound) break;

        lastRound = generateMBAs<Op::And>(MBALen, startPoints, MBACacheCapacity, temp_MBACacheCapacity, allMBAs,
            lastIdx, numOfSamples, d_MBACache, d_temp_MBACache, d_MBAFormula, MBAFormula, "(&)  ", hashSet);
        if (lastRound) break;

        lastRound = generateMBAs<Op::Or>(MBALen, startPoints, MBACacheCapacity, temp_MBACacheCapacity, allMBAs,
            lastIdx, numOfSamples, d_MBACache, d_temp_MBACache, d_MBAFormula, MBAFormula, "(|)  ", hashSet);
        if (lastRound) break;

        lastRound = generateMBAs<Op::Xor>(MBALen, startPoints, MBACacheCapacity, temp_MBACacheCapacity, allMBAs,
            lastIdx, numOfSamples, d_MBACache, d_temp_MBACache, d_MBAFormula, MBAFormula, "(^)  ", hashSet);
        if (lastRound) break;

        lastRound = generateMBAs<Op::LShift>(MBALen, startPoints, MBACacheCapacity, temp_MBACacheCapacity, allMBAs,
            lastIdx, numOfSamples, d_MBACache, d_temp_MBACache, d_MBAFormula, MBAFormula, "(<<) ", hashSet);
        if (lastRound) break;

        lastRound = generateMBAs<Op::RShift>(MBALen, startPoints, MBACacheCapacity, temp_MBACacheCapacity, allMBAs,
            lastIdx, numOfSamples, d_MBACache, d_temp_MBACache, d_MBAFormula, MBAFormula, "(>>) ", hashSet);
        if (lastRound) break;

        lastRound = generateMBAs<Op::Neg>(MBALen, startPoints, MBACacheCapacity, temp_MBACacheCapacity, allMBAs,
            lastIdx, numOfSamples, d_MBACache, d_temp_MBACache, d_MBAFormula, MBAFormula, "(Neg)", hashSet);
        if (lastRound) break;

        lastRound = generateMBAs<Op::Plus>(MBALen, startPoints, MBACacheCapacity, temp_MBACacheCapacity, allMBAs,
            lastIdx, numOfSamples, d_MBACache, d_temp_MBACache, d_MBAFormula, MBAFormula, "(+)  ", hashSet);
        if (lastRound) break;

        lastRound = generateMBAs<Op::Minus>(MBALen, startPoints, MBACacheCapacity, temp_MBACacheCapacity, allMBAs,
            lastIdx, numOfSamples, d_MBACache, d_temp_MBACache, d_MBAFormula, MBAFormula, "(-)  ", hashSet);
        if (lastRound) break;

        lastRound = generateMBAs<Op::Mul>(MBALen, startPoints, MBACacheCapacity, temp_MBACacheCapacity, allMBAs,
            lastIdx, numOfSamples, d_MBACache, d_temp_MBACache, d_MBAFormula, MBAFormula, "(*)  ", hashSet);
        if (lastRound) break;

        lastRound = generateMBAs<Op::Div>(MBALen, startPoints, MBACacheCapacity, temp_MBACacheCapacity, allMBAs,
            lastIdx, numOfSamples, d_MBACache, d_temp_MBACache, d_MBAFormula, MBAFormula, "(/)  ", hashSet);
        if (lastRound) break;

        lastRound = generateMBAs<Op::Mod>(MBALen, startPoints, MBACacheCapacity, temp_MBACacheCapacity, allMBAs,
            lastIdx, numOfSamples, d_MBACache, d_temp_MBACache, d_MBAFormula, MBAFormula, "(%)  ", hashSet);
        if (lastRound) break;

    }

    string output;

    if (MBAFormula[0] != '\0') {
        output = MBAToString(MBAFormula);
    } else {
        output = "Not found !";
    }

    // Cleanup
    cudaFree(d_MBACache);
    cudaFree(d_temp_MBACache);
    cudaFree(d_MBAFormula);

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

    if (atoi(argv[2]) < 1 || atoi(argv[2]) > 50) {
        printf("Argument maxLen = %s should be between 1 and 50", argv[2]);
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

#ifdef MEASUREMENT_MODE
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
    // printf("\nNumber of All LTLs: %lu", allLTLs);
    // printf("\nCost of Final LTL: %d", LTLcost);
    printf("\nRunning Time: %f s", (double)duration * 0.000001);
#endif

    printf("\nMBA: \"%s\"\n", output.c_str());
    return 0;

}