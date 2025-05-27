#include <iostream>
#include <fstream>
#include <stack>
#include "json.hpp"

// #define MEASUREMENT_MODE

using namespace std;
using json = nlohmann::json;

const int maxNumOfVars = 10;
const int maxNumOfSamples = 100;
const int maxFormulaSize = 32;
int numVar;
const char* opStr[8] = { "(~)", "(&)", "(|)", "(^)", "(Neg)", "(+)", "(-)", "(*)" };

__constant__ int d_numVar;
__constant__ uint32_t d_inputData[maxNumOfVars * maxNumOfSamples];
__constant__ uint32_t d_outputData[maxNumOfSamples];
__device__ __constant__ char d_opChar[8] = { '~', '&', '|', '^', '#', '+', '-', '*' };
__constant__ uint64_t d_numForm[maxFormulaSize + 1][9];

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

    numVar = j["0"]["inputs"].size();
    const int numOfSamples = j.size();

    uint32_t* inputData = new uint32_t[numOfSamples * numVar];
    uint32_t* outputData = new uint32_t[numOfSamples];

    int idx = 0;
    for (const auto& sample : j) {
        outputData[idx / numVar] = std::stoul(sample["output"].get<std::string>(), nullptr, 16);
        for (const auto& value : sample["inputs"]) {
            inputData[idx++] = std::stoul(value.get<std::string>(), nullptr, 16);
        }
    }

    return { inputData, outputData, numOfSamples };
}

__device__ bool evaluateRPN(char formula[maxFormulaSize], int size, int numOfSamples) {

    uint32_t stack[32];
    int stackIdx;

    for (int i = 0; i < numOfSamples; ++i) {

        stackIdx = -1;

        for (int j = 0; j < size; ++j) {

            char token = formula[j];

            if (token >= '0' && token <= '9') {
                // Variables
                int var = token - '0';
                stack[++stackIdx] = d_inputData[i * d_numVar + var];
            } else {
                uint32_t left, right;
                if (token == '~' || token == '#') {
                    // Unary operators
                    left = stack[stackIdx];
                    switch (token) {
                    case '~': stack[stackIdx] = ~left; break;
                    case '#': stack[stackIdx] = -left; break;
                    }
                } else {
                    // Binary operators
                    right = stack[stackIdx--];
                    left = stack[stackIdx];
                    switch (token) {
                    case '&': stack[stackIdx] = left & right; break;
                    case '|': stack[stackIdx] = left | right; break;
                    case '^': stack[stackIdx] = left ^ right; break;
                    case '+': stack[stackIdx] = left + right; break;
                    case '-': stack[stackIdx] = left - right; break;
                    case '*': stack[stackIdx] = left * right; break;
                    }
                }
            }

        }

        if (stack[0] != d_outputData[i]) return false;

    }

    return true;

}

struct StackEntry {
    uint64_t n;
    int size;
    int shift;
};

__device__ void printFormula(int size, char formula[maxFormulaSize]) {
    for (int i = 0; i < size; ++i) printf("%c ", formula[i]);
    printf("\n");
}

__device__ void numberToFormula(uint64_t n, int size, char formula[maxFormulaSize]) {

    StackEntry stack[maxFormulaSize];
    int shift = 0;
    stack[0] = { n, size, shift };
    int stackIdx = 0;

    uint64_t opPartsum; int opIdx;
    uint64_t fPartsum; int fIdx;
    uint64_t rightsum;

    while (stackIdx >= 0) {

        StackEntry entry = stack[stackIdx--];
        n = entry.n; size = entry.size; shift = entry.shift;

        if (size == 1) formula[shift] = '0' + n;

        else {

            opPartsum = 0; opIdx = 0;

            // Find the next operator
            while (n >= opPartsum + d_numForm[size][opIdx]) opPartsum += d_numForm[size][opIdx++];
            while (d_numForm[size][opIdx] == 0) opIdx--;
            n -= opPartsum;

            size--;
            formula[shift + size] = d_opChar[opIdx];

            if (opIdx == 0 || opIdx == 4) stack[++stackIdx] = { n, size, shift };

            else {

                fPartsum = 0; fIdx = 1;

                // Operator is binary : Find the sizes of the operands
                while (n >= fPartsum + d_numForm[fIdx][8] * d_numForm[size - fIdx][8]) {
                    fPartsum += d_numForm[fIdx][8] * d_numForm[size - fIdx][8];
                    fIdx++;
                }
                n -= fPartsum;

                rightsum = d_numForm[size - fIdx][8];
                stack[++stackIdx] = { n / rightsum, fIdx, shift };
                stack[++stackIdx] = { n % rightsum, size - fIdx, shift + fIdx };

            }

        }

    }

}

__global__ void processOperator(
    const int numOfSamples, int MBALen,
    uint64_t offset, uint64_t maxTid,
    char* d_MBAFormula)
{

    uint64_t tid = static_cast<uint64_t>(blockDim.x) * static_cast<uint64_t>(blockIdx.x) + static_cast<uint64_t>(threadIdx.x);

    if (tid < maxTid) {

        tid += offset;

        char formula[maxFormulaSize];
        numberToFormula(tid, MBALen, formula);
        bool found = evaluateRPN(formula, MBALen, numOfSamples);

        if (found) for (int i = 0; i < MBALen; ++i) d_MBAFormula[i] = formula[i];

    }

}

void printMatrix(uint64_t* matrix, int m, int n) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) printf("%-12lu ", matrix[i * n + j]);
        printf("\n");
    }
}

uint64_t* generateMatrix(int maxLen) {

    // Last column is the number of formulas of the given size
    uint64_t* numForm = new uint64_t[(maxLen + 1) * 9]();

    // Initialisation with variables and unary operators
    numForm[1 * 9 + 8] = numVar;
    numForm[2 * 9 + 0] = numVar;
    numForm[2 * 9 + 4] = numVar;
    numForm[2 * 9 + 8] = 2 * numVar;

    // Generation
    uint64_t numOfUnaryFormulas;
    uint64_t numOfCommutBinaryFormulas;
    uint64_t numOfNotCommutBinaryFormulas;

    for (int i = 3; i <= maxLen; ++i) {
        numOfUnaryFormulas = numForm[(i - 1) * 9 + 8];
        numOfCommutBinaryFormulas = 0;
        numOfNotCommutBinaryFormulas = 0;
        for (int j = 1; j <= (i - 1) / 2; ++j) {
            numOfCommutBinaryFormulas += numForm[j * 9 + 8] * numForm[(i - j - 1) * 9 + 8];
        }
        for (int j = 1; j <= i - 2; ++j) {
            numOfNotCommutBinaryFormulas += numForm[j * 9 + 8] * numForm[(i - j - 1) * 9 + 8];
        }
        numForm[i * 9] = numOfUnaryFormulas;
        numForm[i * 9 + 1] = numOfCommutBinaryFormulas;
        numForm[i * 9 + 2] = numOfCommutBinaryFormulas;
        numForm[i * 9 + 3] = numOfCommutBinaryFormulas;
        numForm[i * 9 + 4] = numOfUnaryFormulas;
        numForm[i * 9 + 5] = numOfCommutBinaryFormulas;
        numForm[i * 9 + 6] = numOfNotCommutBinaryFormulas;
        numForm[i * 9 + 7] = numOfCommutBinaryFormulas;
        numForm[i * 9 + 8] = 2 * numOfUnaryFormulas + 5 * numOfCommutBinaryFormulas + numOfNotCommutBinaryFormulas;
    }

    printMatrix(numForm, maxLen + 1, 9);
    return numForm;

}

string MBAToString(const char* MBAFormula) {

    stack<string> stack;
    unordered_map<char, string> opToStr = {
        {'~', "~"}, {'&', "&"}, {'|', "|"}, {'^', "^"},
        {'#', "-"}, {'+', "+"}, {'-', "-"}, {'*', "*"},
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

string MBA(
    const int maxLen,
    const int numOfSamples,
    uint32_t* inputData,
    uint32_t* outputData)
{

    // --------------------------------------
    // Memory allocation & Checking variables
    // --------------------------------------

    if (maxLen > maxFormulaSize) {
        printf("This version supports formulas of size at most %d.\n", maxFormulaSize);
        return "see_the_error";
    }

    if (numOfSamples > maxNumOfSamples) {
        printf("This version supports at most %d samples.\n", maxNumOfSamples);
        return "see_the_error";
    }

    if (numVar > maxNumOfVars) {
        printf("This version supports at most %d variables.\n", maxNumOfVars);
        return "see_the_error";
    }

    // Copying number of vars, inputs and outputs into the constant memory
    checkCuda(cudaMemcpyToSymbol(d_numVar, &numVar, sizeof(int)));
    checkCuda(cudaMemcpyToSymbol(d_inputData, inputData, numVar * numOfSamples * sizeof(uint32_t)));
    checkCuda(cudaMemcpyToSymbol(d_outputData, outputData, numOfSamples * sizeof(uint32_t)));

    // Number of generated formulas
    uint64_t allMBAs{};

#ifndef MEASUREMENT_MODE
    printf("Length %-2d | Vars  | CheckedMBAs: %-13lu | ToBeChecked: %-12d \n", 1, allMBAs, numVar);
#endif

    // Checking variables as potential solution
    bool found;
    for (int i = 0; i < numVar; ++i) {
        found = true;
        for (int j = 0; j < numOfSamples; j++) {
            if (!(inputData[j * numVar + i] == outputData[j])) found = false;
        }
        allMBAs++;
        if (found) return "v" + to_string(i);
    }

    // Memory allocation for the potential solution
    char* d_MBAFormula;
    char* MBAFormula = new char[maxFormulaSize]; MBAFormula[0] = '\0';
    checkCuda(cudaMalloc(&d_MBAFormula, maxFormulaSize * sizeof(char)));

    // Number of formulas matrix
    uint64_t* numForm = generateMatrix(maxLen);
    checkCuda(cudaMemcpyToSymbol(d_numForm, numForm, (maxLen + 1) * 9 * sizeof(uint64_t)));

    // ----------------------------
    // Enumeration of the next MBAs
    // ----------------------------

    uint64_t offset;
    uint64_t N;
    uint64_t blockSize;
    bool stop = false;

    for (int MBALen = 2; MBALen <= maxLen && !stop; ++MBALen) {

        offset = 0;

        for (int i = 0; i < 8 && !stop; ++i) {

            N = numForm[MBALen * 9 + i];
            blockSize = (N + 1023) / 1024;
#ifndef MEASUREMENT_MODE
            printf("Length %-2d | %-5s | CheckedMBAs: %-13lu | ToBeChecked: %-12lu \n", MBALen, opStr[i], allMBAs, N);
#endif
            if (N > 0) processOperator << <blockSize, 1024 >> > (numOfSamples, MBALen, offset, N, d_MBAFormula);
            offset += N; allMBAs += N;

            checkCuda(cudaPeekAtLastError());
            checkCuda(cudaMemcpy(MBAFormula, d_MBAFormula, maxFormulaSize * sizeof(char), cudaMemcpyDeviceToHost));
            if (MBAFormula[0] != '\0') stop = true;

        }

    }

    // --------------------------------
    // Returning the solution & Cleanup
    // --------------------------------

    string output;

    if (MBAFormula[0] != '\0') {
        output = MBAToString(MBAFormula);
    } else {
        output = "Not found !";
    }

    delete[] numForm;
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
        printf("Argument maxLen = %s should be between 1 and %d", argv[2], maxFormulaSize);
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