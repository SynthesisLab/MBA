#include <iostream>
#include <fstream>
#include <memory>
#include <vector>
#include <atomic>
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
   vector<vector<shared_ptr<Formula>>> formulas(n + 1);

   // Initialization
   for (int i = 0; i < numVar; ++i) {
      std::shared_ptr<Formula> formula = make_shared<Formula>(Variable(i));
      formulas[1].push_back(formula);

      if (testFormula(*formula, examples)) {
         formula->printFormula();
         return;
      }
   }

   // Generation
   atomic<bool> found(false);
   for (int size = 2; size <= n; ++size) {
      if (found) break;
      cout << "Starting generation of formulas of size " << size << endl;

      // Unary operators
#pragma omp parallel for
      for (const std::shared_ptr<Formula>& formula : formulas[size - 1]) {
         if (found) continue;
         vector<UnOp> ops = { UnOp::Not };
         for (UnOp op : ops) {
            std::shared_ptr<Formula> newFormula = make_shared<Formula>(UnaryOperation(op, formula));
#pragma omp critical
            formulas[size].push_back(newFormula);

            if (testFormula(*newFormula, examples)) {
#pragma omp critical
               newFormula->printFormula();
               found = true;
#pragma omp flush(found)
            }
         }
      }

      // Binary operators
      for (int k = 1; k <= size - 2; ++k) {
#pragma omp parallel for collapse(2)
         for (const std::shared_ptr<Formula>& left : formulas[k]) {
            for (const std::shared_ptr<Formula>& right : formulas[size - k - 1]) {
               if (found) continue;
               vector<BinOp> ops = { BinOp::Plus, BinOp::Minus, BinOp::Mult, BinOp::And, BinOp::Or, BinOp::Xor };
               for (BinOp op : ops) {
                  std::shared_ptr<Formula> newFormula = make_shared<Formula>(BinaryOperation(op, left, right));
#pragma omp critical
                  formulas[size].push_back(newFormula);

                  if (testFormula(*newFormula, examples)) {
#pragma omp critical
                     newFormula->printFormula();
                     found = true;
#pragma omp flush(found)
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
   generateFormulas(11, examples);
   auto end = std::chrono::high_resolution_clock::now();
   std::chrono::duration<double> duration = end - start;
   cout << "Time elapsed: " << duration.count() << " seconds" << endl;
   return 0;
}