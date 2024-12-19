#include <iostream>
#include <fstream>
#include <memory>
#include <set>
#include <vector>
#include <chrono>
#include "formula.h"
#include "read.h"

using namespace std;

// Test if a formula satisfies all examples
bool testFormula(const Formula& formula, const vector<Example>& examples) {
   for (const Example& example : examples) {
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
      std::shared_ptr<Formula> formula = make_shared<Formula>(Variable(i));
      formulas[1].insert(formula);

      if (testFormula(*formula, examples)) {
         formula->printFormula();
         return;
      }
   }

   // Generation
   for (int size = 2; size <= n; ++size) {

      // Unary operators
      for (const std::shared_ptr<Formula>& formula : formulas[size - 1]) {
         vector<UnOp> ops = { UnOp::Not };
         for (UnOp op : ops) {
            std::shared_ptr<Formula> newFormula = make_shared<Formula>(UnaryOperation(op, formula));
            formulas[size].insert(newFormula);

            if (testFormula(*newFormula, examples)) {
               newFormula->printFormula();
               return;
            }
         }
      }

      // Binary operators
      for (int k = 1; k <= size - 2; ++k) {
         for (const std::shared_ptr<Formula>& left : formulas[k]) {
            for (const std::shared_ptr<Formula>& right : formulas[size - k - 1]) {
               vector<BinOp> ops = { BinOp::Plus, BinOp::Minus, BinOp::Mult, BinOp::And, BinOp::Or, BinOp::Xor };
               for (BinOp op : ops) {
                  std::shared_ptr<Formula> newFormula = make_shared<Formula>(BinaryOperation(op, left, right));
                  formulas[size].insert(newFormula);

                  if (testFormula(*newFormula, examples)) {
                     newFormula->printFormula();
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
   const string filename = "../benchmarks/test.json";
   vector<Example> examples = readExamples(filename);

   auto start = std::chrono::high_resolution_clock::now();
   generateFormulas(10, examples);
   auto end = std::chrono::high_resolution_clock::now();
   std::chrono::duration<double> duration = end - start;
   cout << "Time elapsed: " << duration.count() << " seconds" << endl;
   return 0;
}