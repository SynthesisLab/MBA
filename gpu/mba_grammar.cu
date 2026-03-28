#include <iostream>
#include <fstream>
#include <sstream>
#include <stack>
#include <string>
#include <vector>
#include "json.hpp"

using namespace std;
using json = nlohmann::json;

const int maxNumOfVars = 10;
const int maxNumOfSamples = 100;
const int maxFormulaSize = 16;
const int maxNumOfStates = 1000;
const int maxNumOfRules = 10000;

int numVar;
int numOfSamples;
int numS;
int numR;

__constant__ int d_numVar;
__constant__ int d_numOfSamples;
__constant__ int d_numS;
__constant__ int d_numR;
__constant__ uint32_t d_inputData[maxNumOfVars * maxNumOfSamples];
__constant__ uint32_t d_outputData[maxNumOfSamples];
__constant__ int d_offsets[maxNumOfStates + 1];

// CUDA error checking
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

// Read JSON file
tuple<uint32_t*, uint32_t*> readJsonFile(const string& filename) {

    ifstream file(filename);
    if (!file) {
        cerr << "Error: Could not open file " << filename << endl;
        return { nullptr, nullptr };
    }

    json j;
    file >> j;

    numVar = j["0"]["inputs"].size();
    numOfSamples = j.size();

    uint32_t* inputData = new uint32_t[numOfSamples * numVar];
    uint32_t* outputData = new uint32_t[numOfSamples];

    int idx = 0;
    for (const auto& sample : j) {
        outputData[idx / numVar] = stoul(sample["output"].get<string>(), nullptr, 16);
        for (const auto& value : sample["inputs"]) {
            inputData[idx++] = stoul(value.get<string>(), nullptr, 16);
        }
    }

    return { inputData, outputData };
}

// Read grape file
struct Rule {
    int state;
    char op;
    int operands[3];
    int arity;

    __host__ __device__
        Rule() : state(-1), op('\0'), operands{ -1, -1 }, arity(-1) {}

    __host__ __device__
        Rule(int s, char op_char, const int* next, int ar) : state(s), op(op_char), arity(ar) {
        operands[0] = (ar > 0) ? next[0] : -1;
        operands[1] = (ar > 1) ? next[1] : -1;
    }

    string to_string() const {
        ostringstream oss;
        oss << "S" << state << " -> ";
        if (arity == 1) oss << op << "S" << operands[0];
        else if (arity == 2) oss << "S" << operands[0] << " " << op << " S" << operands[1];
        else oss << "INVALID";
        return oss.str();
    }
};

int* countRules(const string& filename, Rule* rules) {

    ifstream file(filename);
    string line;

    getline(file, line);
    getline(file, line);
    if (!getline(file, line)) {
        cerr << "Erreur : Ligne 'states' manquante\n";
        return nullptr;
    }

    int max_state_index = -1;
    if (line.rfind("states:", 0) == 0) {
        istringstream iss(line.substr(7));
        string token;
        while (getline(iss, token, ',')) {
            if (token.rfind("S", 0) == 0) {
                int num = stoi(token.substr(1));
                max_state_index = max(max_state_index, num);
            }
        }
    } else {
        cerr << "Erreur : Format inattendu pour la ligne 'states'\n";
        return nullptr;
    }

    int* rule_counts = new int[max_state_index + 1];
    memset(rule_counts, 0, (max_state_index + 1) * sizeof(int));
    numS = max_state_index + 1;
    vector<Rule> unord_rules;

    while (getline(file, line)) {

        istringstream iss(line);
        string from_state;
        string op_str;

        if (getline(iss, from_state, ',')) {
            if (from_state.rfind("S", 0) == 0) {
                int index = stoi(from_state.substr(1));
                if (index <= max_state_index) {

                    if (getline(iss, op_str, ',')) {

                        if (op_str.size() != 1) {
                            cerr << "Erreur : Opérateur '" << op_str << "' non supporté (doit être un seul caractère)\n";
                            return nullptr;
                        }
                        char op = op_str[0];

                        int operands[2] = { -1, -1 };
                        int arity = 0;
                        string operand;
                        while (arity < 2 && getline(iss, operand, ',')) {
                            if (operand.rfind("S", 0) == 0) operands[arity++] = stoi(operand.substr(1));
                        }
                        if (getline(iss, operand, ',')) {
                            cerr << "Erreur : Arité > 2 non supportée\n";
                            return nullptr;
                        }

                        Rule rule(index, op, operands, arity);
                        unord_rules.push_back(rule);
                        rule_counts[index]++;

                    }

                }
            }
        }

    }

    numR = unord_rules.size();
    int* offsets = new int[numS + 1];
    offsets[0] = 0;
    for (int i = 1; i <= numS; ++i) offsets[i] = offsets[i - 1] + rule_counts[i - 1];

    int* rules_placed = new int[numS];
    memset(rules_placed, 0, numS * sizeof(int));
    for (const auto& rule : unord_rules) {
        int state = rule.state;
        int pos = offsets[state] + rules_placed[state];
        rules[pos] = rule;
        rules_placed[state]++;
    }

    delete[] rule_counts;
    delete[] rules_placed;
    return offsets;

}

// Compute number of formulas per state and rule
void computeMatrices(
    int maxLen, Rule* rules, int* offsets,
    uint64_t* numSForm, uint64_t* numRForm)
{

    for (int s = 0; s < numS; ++s) {
        for (int r = offsets[s]; r < offsets[s + 1]; ++r) {
            const Rule& rule = rules[r];
            if (rule.arity == 0) {
                numRForm[numR + r] = 1;
                numSForm[numS + s] += 1;
            }
        }
    }

    for (int size = 2; size <= maxLen; ++size) {
        for (int s = 0; s < numS; ++s) {
            for (int r = offsets[s]; r < offsets[s + 1]; ++r) {
                const Rule& rule = rules[r];

                if (rule.arity == 1) {
                    uint64_t num = numSForm[(size - 1) * numS + rule.operands[0]];
                    numRForm[size * numR + r] = num;
                    numSForm[size * numS + s] += num;
                }

                if (rule.arity == 2) {
                    for (int i = 1; i < size - 1; ++i) {
                        uint64_t lNum = numSForm[i * numS + rule.operands[0]];
                        uint64_t rNum = numSForm[(size - 1 - i) * numS + rule.operands[1]];
                        numRForm[size * numR + r] += lNum * rNum;
                    }
                    numSForm[size * numS + s] += numRForm[size * numR + r];
                }

                if (rule.arity == 3) {
                    for (int i = 1; i < size - 2; ++i) {
                        for (int j = 1; j < size - i - 1; ++j) {
                            uint64_t lNum = numSForm[i * numS + rule.operands[0]];
                            uint64_t mNum = numSForm[j * numS + rule.operands[1]];
                            uint64_t rNum = numSForm[(size - 2 - i - j) * numS + rule.operands[2]];
                            numRForm[size * numR + r] += lNum * mNum * rNum;
                        }
                    }
                    numSForm[size * numS + s] += numRForm[size * numR + r];
                }

            }
        }
    }

}

// Reconstruct formula from a number
struct StackEntry {
    uint64_t n;
    int ruleOp;
    int size;
    int shift;
};

__device__ void printFormula(int size, char formula[maxFormulaSize]) {
    for (int i = 0; i < size; ++i) printf("%c ", formula[i]);
    printf("\n");
}

__device__ void numberToFormula(
    uint64_t n, int size, char formula[maxFormulaSize],
    Rule* d_rules, int ruleOp,
    uint64_t* d_numSForm, uint64_t* d_numRForm) {

    StackEntry stack[maxFormulaSize];
    int shift = 0;
    stack[0] = { n, ruleOp, size, shift };
    int stackIdx = 0;

    while (stackIdx >= 0) {

        StackEntry entry = stack[stackIdx--];
        n = entry.n; ruleOp = entry.ruleOp; size = entry.size; shift = entry.shift;

        if (size == 1) formula[shift] = '0' + n;

        else {

            // Find the rule
            uint64_t partSum = 0;
            int ruleIdx = d_offsets[ruleOp];
            while (n >= partSum + d_numRForm[size * d_numR + ruleIdx]) partSum += d_numRForm[size * d_numR + ruleIdx++];
            n -= partSum;

            size--;
            Rule rule = d_rules[ruleIdx];
            formula[shift + size] = rule.op;

            if (rule.arity == 1) stack[++stackIdx] = { n, rule.operands[0], size, shift };

            if (rule.arity == 2) {

                partSum = 0; int lIdx = 1; int rIdx = size - 1;
                int left = rule.operands[0]; int right = rule.operands[1];

                // Find the sizes of the operands
                while (n >= partSum + d_numSForm[lIdx * d_numS + left] * d_numSForm[rIdx * d_numS + right])
                    partSum += d_numSForm[lIdx++ * d_numS + left] * d_numSForm[rIdx-- * d_numS + right];
                n -= partSum;

                uint64_t rSum = d_numSForm[rIdx * d_numS + right];
                stack[++stackIdx] = { n / rSum, left, lIdx, shift };
                stack[++stackIdx] = { n % rSum, right, rIdx, shift + lIdx };

            }

            if (rule.arity == 3) {

                partSum = 0; int lIdx = 1; int mIdx = 1; int rIdx = size - 2;
                int left = rule.operands[0]; int mid = rule.operands[1]; int right = rule.operands[2];

                // Find the sizes of the operands
                while (n >= partSum + d_numSForm[lIdx * d_numS + left] * d_numSForm[mIdx * d_numS + mid] * d_numSForm[rIdx * d_numS + right]) {
                    partSum += d_numSForm[lIdx * d_numS + left] * d_numSForm[mIdx++ * d_numS + mid] * d_numSForm[rIdx-- * d_numS + right];
                    if (rIdx == 0) { lIdx++; mIdx = 1; rIdx = size - lIdx - 1; }
                }
                n -= partSum;

                uint64_t mSum = d_numSForm[mIdx * d_numS + mid];
                uint64_t rSum = d_numSForm[rIdx * d_numS + right];
                stack[++stackIdx] = { n / (mSum * rSum) , left, lIdx, shift };
                stack[++stackIdx] = { (n / rSum) % mSum, mid, mIdx, shift + lIdx };
                stack[++stackIdx] = { n % rSum, right, rIdx, shift + lIdx + mIdx };

            }

        }

    }

}

// Evaluation
__device__ bool evaluateRPN(char formula[maxFormulaSize], int size) {

    uint32_t stack[32];
    int stackIdx;

    for (int i = 0; i < d_numOfSamples; ++i) {

        stackIdx = -1;

        for (int j = 0; j < size; ++j) {

            char token = formula[j];

            if (token >= '0' && token <= '9') {
                // Variables
                int var = token - '0';
                stack[++stackIdx] = d_inputData[i * d_numVar + var];
            } else {
                uint32_t left, right;
                if (token == '~') {
                    // Unary operators
                    left = stack[stackIdx];
                    stack[stackIdx] = ~left;
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

// Generate and check formulas
__global__ void processOperator(
    int MBALen,
    uint64_t offset, uint64_t maxTid,
    Rule* d_rules, int ruleOp,
    uint64_t* d_numSForm, uint64_t* d_numRForm,
    char* d_MBAFormula)
{

    uint64_t tid = static_cast<uint64_t>(blockDim.x) * static_cast<uint64_t>(blockIdx.x) + static_cast<uint64_t>(threadIdx.x);

    if (tid < maxTid) {

        tid += offset;

        char formula[maxFormulaSize];
        numberToFormula(tid, MBALen, formula, d_rules, ruleOp, d_numSForm, d_numRForm);
        if (tid - offset == 0) printFormula(MBALen, formula);
        bool found = evaluateRPN(formula, MBALen);
        if (found) for (int i = 0; i < MBALen; ++i) d_MBAFormula[i] = formula[i];

    }

}

void printMatrix(uint64_t* matrix, int m, int n) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) printf("%-12lu ", matrix[i * n + j]);
        printf("\n");
    }
}


// Convert MBA formula to string
string MBAToString(const char* MBAFormula) {

    stack<string> stack;

    for (int i = 0; MBAFormula[i] != '\0'; ++i) {

        char token = MBAFormula[i];

        if (token >= '0' && token <= '9') {
            stack.push("v" + string(1, token));
        } else {
            if (token == '~') {
                if (stack.empty()) return "Error : Incorrect formula.";
                string operand = stack.top(); stack.pop();
                stack.push("(" + string(1, token) + operand + ")");
            } else {
                if (stack.size() < 2) return "Error : Incorrect formula.";
                string right = stack.top(); stack.pop();
                string left = stack.top(); stack.pop();
                stack.push("(" + left + " " + string(1, token) + " " + right + ")");
            }
        }

    }

    if (stack.size() != 1) return "Error : Incorrect formula.";
    return stack.top();

}

// Enumerate formulas
string MBA(
    const int maxLen,
    Rule* rules, int* offsets,
    uint64_t* numSForm, uint64_t* numRForm,
    uint32_t* inputData, uint32_t* outputData) {

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

    if (numS > maxNumOfStates) {
        printf("This version supports at most %d states.\n", maxNumOfStates);
        return "see_the_error";
    }

    if (numR > maxNumOfRules) {
        printf("This version supports at most %d rules.\n", maxNumOfRules);
        return "see_the_error";
    }

    // Copying number of vars and samples, states, offsets, and inputs and outputs into the constant memory
    checkCuda(cudaMemcpyToSymbol(d_numVar, &numVar, sizeof(int)));
    checkCuda(cudaMemcpyToSymbol(d_numOfSamples, &numOfSamples, sizeof(int)));
    checkCuda(cudaMemcpyToSymbol(d_numS, &numS, sizeof(int)));
    checkCuda(cudaMemcpyToSymbol(d_numR, &numR, sizeof(int)));
    checkCuda(cudaMemcpyToSymbol(d_offsets, offsets, (numS + 1) * sizeof(int)));
    checkCuda(cudaMemcpyToSymbol(d_inputData, inputData, numVar * numOfSamples * sizeof(uint32_t)));
    checkCuda(cudaMemcpyToSymbol(d_outputData, outputData, numOfSamples * sizeof(uint32_t)));

    // Number of generated formulas
    uint64_t allMBAs{};

    printf("Length %-2d | Vars                | Checked: %-13lu | Checking: %-12d \n", 1, allMBAs, numVar);

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

    // Memory allocation for the rules
    Rule* d_rules;
    checkCuda(cudaMalloc(&d_rules, numR * sizeof(Rule)));
    checkCuda(cudaMemcpy(d_rules, rules, numR * sizeof(Rule), cudaMemcpyHostToDevice));

    // Memory allocation for the matrices
    uint64_t* d_numRForm; uint64_t* d_numSForm;
    checkCuda(cudaMalloc(&d_numRForm, (maxLen + 1) * numR * sizeof(uint64_t)));
    checkCuda(cudaMalloc(&d_numSForm, (maxLen + 1) * numS * sizeof(uint64_t)));
    checkCuda(cudaMemcpy(d_numRForm, numRForm, (maxLen + 1) * numR * sizeof(uint64_t), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_numSForm, numSForm, (maxLen + 1) * numS * sizeof(uint64_t), cudaMemcpyHostToDevice));

    // ----------------------------
    // Enumeration of the next MBAs
    // ----------------------------

    uint64_t offset;
    uint64_t N;
    uint64_t blockSize;
    bool stop = false;

    for (int MBALen = 2; MBALen <= maxLen && !stop; ++MBALen) {

        for (int i = 0; i < numS && !stop; ++i) {

            offset = 0;

            for (int j = offsets[i]; j < offsets[i + 1] && !stop; ++j) {

                N = numRForm[MBALen * numR + j];
                blockSize = (N + 1023) / 1024;
                if (N > 0) {
                    printf("Length %-2d | %-19s | Checked: %-13lu | Checking: %-12lu \n", MBALen, rules[j].to_string().c_str(), allMBAs, N);
                    processOperator << <blockSize, 1024 >> > (MBALen, offset, N, d_rules, i, d_numSForm, d_numRForm, d_MBAFormula);
                }
                offset += N; allMBAs += N;

                checkCuda(cudaPeekAtLastError());
                checkCuda(cudaMemcpy(MBAFormula, d_MBAFormula, maxFormulaSize * sizeof(char), cudaMemcpyDeviceToHost));
                if (MBAFormula[0] != '\0') stop = true;

            }

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

    cudaFree(d_numRForm); cudaFree(d_numSForm); cudaFree(d_MBAFormula);
    return output;

}

int main(int argc, char* argv[]) {

    // -----------------
    // Reading the input
    // -----------------

    if (argc != 4) {
        printf("Arguments should be in the form of\n");
        printf("----------------------------------------------------------------------\n");
        printf("%s <input_file_address> <grape_file_address> <maxLen>\n", argv[0]);
        printf("----------------------------------------------------------------------\n");
        return 0;
    }

    int maxLen = atoi(argv[3]);
    if (maxLen < 1 || maxLen > maxFormulaSize) {
        printf("Argument maxLen = %s should be between 1 and %d\n", argv[3], maxFormulaSize);
        return 0;
    }

    string inputFile = argv[1];
    auto [inputData, outputData] = readJsonFile(inputFile);

    string grapeFile = argv[2];
    Rule* rules = new Rule[maxNumOfRules];
    int* offsets = countRules(grapeFile, rules);

    uint64_t* numSForm = new uint64_t[(maxLen + 1) * numS];
    uint64_t* numRForm = new uint64_t[(maxLen + 1) * numR];

    memset(numSForm, 0, (maxLen + 1) * numS * sizeof(uint64_t));
    memset(numRForm, 0, (maxLen + 1) * numR * sizeof(uint64_t));

    computeMatrices(maxLen, rules, offsets, numSForm, numRForm);

    string output = MBA(maxLen, rules, offsets, numSForm, numRForm, inputData, outputData);
    if (output == "see_the_error") return 0;
    printf("\nMBA: \"%s\"\n", output.c_str());

    // Nettoyage
    delete[] rules;
    delete[] offsets;
    delete[] numRForm;
    delete[] numSForm;

    return 0;

}