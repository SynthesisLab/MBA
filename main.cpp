#include <iostream>
#include <fstream>
#include <set>
#include "formula.h"
#include "json.hpp"

using namespace std;
using json = nlohmann::json;

int numVar;

// Generate formulas until size n
vector<set<shared_ptr<Formula>>> generateFormulas(int n) {
   vector<set<shared_ptr<Formula>>> formulas(n + 1);

   // Formulas of size 1
   for (int i = 0; i < numVar; ++i) {
      formulas[1].insert(make_shared<Formula>(Variable(i)));
   }

   // Generation
   for (int size = 2; size <= n; ++size) {

      // Unary operators
      for (const auto& formula : formulas[size - 1]) {
         formulas[size].insert(make_shared<Formula>(UnaryOperation(UnOp::Not, formula)));
      }

      // Binary operators
      for (int k = 1; k <= size - 2; ++k) {
         for (const auto& left : formulas[k]) {
            for (const auto& right : formulas[size - k - 1]) {
               formulas[size].insert(make_shared<Formula>(BinaryOperation(BinOp::Plus, left, right)));
               formulas[size].insert(make_shared<Formula>(BinaryOperation(BinOp::Minus, left, right)));
               formulas[size].insert(make_shared<Formula>(BinaryOperation(BinOp::Mult, left, right)));
               formulas[size].insert(make_shared<Formula>(BinaryOperation(BinOp::And, left, right)));
               formulas[size].insert(make_shared<Formula>(BinaryOperation(BinOp::Or, left, right)));
               formulas[size].insert(make_shared<Formula>(BinaryOperation(BinOp::Xor, left, right)));
            }
         }
      }
   }

   return formulas;
}

// Print generated formulas
void printFormulas(const vector<set<shared_ptr<Formula>>>& formulas) {
   for (size_t i = 1; i < formulas.size(); ++i) {
      cout << "Formules de taille " << i << " :\n";
      for (const auto& formula : formulas[i]) {
         formula->printFormula();
      }
   }
}

// Antecedents & Image pairs
class Example
{
public:
   vector<uint64_t> inputs; // Antecedents
   uint64_t output; // Image
};

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

// Check whether a formula satisfies all examples
bool testFormulaOnExamples(const Formula& formula, const vector<Example>& examples) {
   for (const auto& example : examples) {
      uint64_t result = formula.evaluate(example.inputs);
      if (result != example.output) {
         return false;
      }
   }
   return true;
}


int main() {

   const string filename = "example_8_and_16bits.json";
   vector<Example> examples = readExamples(filename);

   auto formulas = generateFormulas(3);
   printFormulas(formulas);

   // try {
   //    vector<Example> examples = readExamples(filename);

   //    // Display the examples
   //    for (const auto& example : examples) {
   //       cout << "Inputs: ";
   //       for (const auto& input : example.inputs) {
   //          cout << hex << input << " "; // Print inputs in hex
   //       }
   //       cout << "Output: " << hex << example.output << endl; // Print output in hex
   //    }
   // }
   // catch (const exception& e) {
   //    cerr << "Error: " << e.what() << endl;
   // }

   for (const auto& formulaSet : formulas) {
      for (const auto& formula : formulaSet) {
         if (testFormulaOnExamples(*formula, examples)) {
            cout << "Found a matching formula:\n";
            formula->printFormula();
            break;
         }
      }
   }

   return 0;
}