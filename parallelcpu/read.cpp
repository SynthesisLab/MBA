#include <fstream>
#include "json.hpp"
#include "read.h"

using namespace std;
using json = nlohmann::json;

int numVar;

// Read JSON file and extract examples
vector<Example> readExamples(const string& filename) {
    vector<Example> examples;

    // Read the JSON file
    ifstream inputFile(filename);
    if (!inputFile) {
        throw runtime_error("Could not open the file!");
    }

    json j;
    inputFile >> j;

    // Extract the initial example
    Example initialExample;
    const auto& initialInputs = j["initial"]["inputs"];
    const auto& initialOutput = j["initial"]["outputs"].begin().value(); // Assuming one output

    // Set numVar based on the number of inputs
    numVar = initialInputs.size();

    // Extract initial inputs
    for (const auto& input : initialInputs.items()) {
        initialExample.inputs.push_back(stoull(input.value()["value"].get<string>(), nullptr, 16));
    }

    // Extract initial output
    initialExample.output = stoull(initialOutput["value"].get<string>(), nullptr, 16);

    // Add the initial example to the vector
    examples.push_back(initialExample);

    // Extract sampling examples
    for (const auto& item : j["sampling"].items()) {
        Example example;
        const auto& inputs = item.value()["inputs"];
        const auto& output = item.value()["outputs"].begin().value(); // Assuming one output

        // Extract inputs
        for (const auto& input : inputs.items()) {
            example.inputs.push_back(stoull(input.value()["value"].get<string>(), nullptr, 16));
        }

        // Extract output
        example.output = stoull(output["value"].get<string>(), nullptr, 16);

        // Add the example to the vector
        examples.push_back(example);
    }

    return examples;
}