#include <iostream>
#include <fstream>
#include <memory>
#include <set>
#include <vector>
#include "formula.h"
#include "read.h"

using namespace std;

// Test if a formula satisfies all examples
bool testFormula(const Formula& formula, const vector<Example>& examples) {
   for (const auto& example : examples) {
      uint32_t result = formula.evaluate(example.inputs);
      if (result != example.output) {
         return false;
      }
   }
   return true;
}

// Generate formulas until size n
void generateFormulas(int n, const vector<Example>& examples) {
   vector<set<shared_ptr<Formula>>> formulas(n + 1);

   // Initialization
   for (int i = 0; i < numVar; ++i) {
      auto formula = make_shared<Formula>(Variable(i));
      formulas[1].insert(formula);
      formula->printFormula();

      if (testFormula(*formula, examples)) {
         cout << "Found !\n";
         return;
      }
   }

   // Generation
   for (int size = 2; size <= n; ++size) {

      // Unary operators
      for (const auto& formula : formulas[size - 1]) {
         vector<UnOp> ops = { UnOp::Not };
         for (auto op : ops) {
            auto newFormula = make_shared<Formula>(UnaryOperation(op, formula));
            formulas[size].insert(newFormula);
            newFormula->printFormula();

            if (testFormula(*newFormula, examples)) {
               cout << "Found !\n";
               return;
            }
         }
      }

      // Binary operators
      for (int k = 1; k <= size - 2; ++k) {
         for (const auto& left : formulas[k]) {
            for (const auto& right : formulas[size - k - 1]) {
               vector<BinOp> ops = { BinOp::Plus, BinOp::Minus, BinOp::Mult, BinOp::And, BinOp::Or, BinOp::Xor };
               for (auto op : ops) {
                  auto newFormula = make_shared<Formula>(BinaryOperation(op, left, right));
                  formulas[size].insert(newFormula);
                  newFormula->printFormula();

                  if (testFormula(*newFormula, examples)) {
                     cout << "Found !\n";
                     return;
                  }
               }
            }
         }
      }
   }

   return;
}

int main() {

   const string filename = "samples/1.json";
   vector<Example> examples = readExamples(filename);

   generateFormulas(7, examples);
   return 0;

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
}