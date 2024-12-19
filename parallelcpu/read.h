#ifndef READ_H
#define READ_H

#include <iostream>
#include <vector>

using namespace std;

// Number of antecedents
extern int numVar;

// Antecedents & Image pairs
class Example
{
public:
    vector<uint32_t> inputs;
    uint32_t output;
};

// Read json file
vector<Example> readExamples(const string& filename);

#endif