#include <iostream>
#include <fstream>
#include <vector>
#include <set>
#include <memory>
#include <variant>
#include <string>
#include "json.hpp"

using namespace std;
using json = nlohmann::json;

int numVar = 2;

struct Formula;

// Unary operators
enum class UnOp { Not };

// Binary operators
enum class BinOp { Plus, Minus, Mult, And, Or, Xor };

// Variables
struct Variable
{
   int var;

   Variable(int v) : var(v) {}
};

// Unary operation
struct UnaryOperation
{
   UnOp op;
   std::shared_ptr<Formula> operand;

   UnaryOperation(UnOp op, std::shared_ptr<Formula> operand) : op(op), operand(operand) {}
};

// Binary operation
struct BinaryOperation
{
   BinOp op;
   std::shared_ptr<Formula> left;
   std::shared_ptr<Formula> right;

   BinaryOperation(BinOp op, std::shared_ptr<Formula> left, std::shared_ptr<Formula> right) : op(op), left(left), right(right) {}
};

// Formula
struct Formula
{
   std::variant<Variable, BinaryOperation, UnaryOperation> expr;

   Formula(Variable var) : expr(var) {}
   Formula(BinaryOperation binOp) : expr(binOp) {}
   Formula(UnaryOperation unOp) : expr(unOp) {}
};

// String of unary operator
std::string to_string(UnOp op)
{
   switch (op)
   {
   case UnOp::Not: return "~";
   default: return "?";
   }
}

// String of binary operator
std::string to_string(BinOp op)
{
   switch (op)
   {
   case BinOp::Plus: return "+";
   case BinOp::Minus: return "-";
   case BinOp::Mult: return "*";
   case BinOp::And: return "&";
   case BinOp::Or: return "|";
   case BinOp::Xor: return "^";
   default: return "?";
   }
}

// Generate formulas until size n
std::vector<std::set<std::shared_ptr<Formula>>> generateFormulas(int n) {
   std::vector<std::set<std::shared_ptr<Formula>>> formulas(n + 1);

   // Formulas of size 1
   for (int i = 0; i < numVar; ++i) {
      formulas[1].insert(std::make_shared<Formula>(Variable(i)));
   }

   // Generation
   for (int size = 2; size <= n; ++size) {

      // Unary operators
      for (const auto& formula : formulas[size - 1]) {
         formulas[size].insert(std::make_shared<Formula>(UnaryOperation(UnOp::Not, formula)));
      }

      // Binary operators
      for (int k = 1; k <= size - 2; ++k) {
         for (const auto& left : formulas[k]) {
            for (const auto& right : formulas[size - k - 1]) {
               formulas[size].insert(std::make_shared<Formula>(BinaryOperation(BinOp::Plus, left, right)));
               formulas[size].insert(std::make_shared<Formula>(BinaryOperation(BinOp::Minus, left, right)));
               formulas[size].insert(std::make_shared<Formula>(BinaryOperation(BinOp::Mult, left, right)));
               formulas[size].insert(std::make_shared<Formula>(BinaryOperation(BinOp::And, left, right)));
               formulas[size].insert(std::make_shared<Formula>(BinaryOperation(BinOp::Or, left, right)));
               formulas[size].insert(std::make_shared<Formula>(BinaryOperation(BinOp::Xor, left, right)));
            }
         }
      }
   }

   return formulas;
}

// Print formula
void printFormula(const Formula& formula)
{
   std::visit([](auto&& arg)
      {
         using T = std::decay_t<decltype(arg)>;
         if constexpr (std::is_same_v<T, Variable>) {
            std::cout << "Var(" << arg.var << ")";
         }
         else if constexpr (std::is_same_v<T, BinaryOperation>) {
            std::cout << "(";
            printFormula(*arg.left);
            std::cout << " " << to_string(arg.op) << " ";
            printFormula(*arg.right);
            std::cout << ")";
         }
         else if constexpr (std::is_same_v<T, UnaryOperation>) {
            std::cout << "(" << to_string(arg.op) << " ";
            printFormula(*arg.operand);
            std::cout << ")";
         } }, formula.expr);
}

// Print generated formulas
void printFormulas(const std::vector<std::set<std::shared_ptr<Formula>>>& formulas) {
   for (size_t i = 1; i < formulas.size(); ++i) {
      std::cout << "Formules de taille " << i << " :\n";
      for (const auto& formula : formulas[i]) {
         printFormula(*formula);
         std::cout << "\n";
      }
   }
}

// Antecedents & Image pairs
struct Example
{
   std::vector<uint64_t> inputs; // Antecedents
   uint64_t output; // Image
};

// Read JSON file and extract examples
std::vector<Example> readExamples(const std::string& filename) {
   std::vector<Example> examples;

   // Read the JSON file
   std::ifstream inputFile(filename);
   if (!inputFile) {
      throw std::runtime_error("Could not open the file!");
   }

   json j;
   inputFile >> j;

   // Extract the initial example
   Example initialExample;
   const auto& initialInputs = j["initial"]["inputs"];
   const auto& initialOutput = j["initial"]["outputs"].begin().value(); // Assuming one output

   // Extract initial inputs
   for (const auto& input : initialInputs.items()) {
      initialExample.inputs.push_back(std::stoull(input.value()["value"].get<std::string>(), nullptr, 16));
   }

   // Extract initial output
   initialExample.output = std::stoull(initialOutput["value"].get<std::string>(), nullptr, 16);

   // Add the initial example to the vector
   examples.push_back(initialExample);

   // Extract sampling examples
   for (const auto& item : j["sampling"].items()) {
      Example example;
      const auto& inputs = item.value()["inputs"];
      const auto& output = item.value()["outputs"].begin().value(); // Assuming one output

      // Extract inputs
      for (const auto& input : inputs.items()) {
         example.inputs.push_back(std::stoull(input.value()["value"].get<std::string>(), nullptr, 16));
      }

      // Extract output
      example.output = std::stoull(output["value"].get<std::string>(), nullptr, 16);

      // Add the example to the vector
      examples.push_back(example);
   }

   return examples;
}

// Evaluate formula on inputs
uint64_t evaluateFormula(const Formula& formula, const std::vector<uint64_t>& inputs) {
   return std::visit([&inputs](auto&& arg) -> uint64_t {
      using T = std::decay_t<decltype(arg)>;

      if constexpr (std::is_same_v<T, Variable>) {
         return inputs[arg.var];
      }
      else if constexpr (std::is_same_v<T, UnaryOperation>) {
         uint64_t operandValue = evaluateFormula(*arg.operand, inputs);
         switch (arg.op) {
         case UnOp::Not: return ~operandValue;
         default: throw std::runtime_error("Unsupported unary operation");
         }
      }
      else if constexpr (std::is_same_v<T, BinaryOperation>) {
         uint64_t leftValue = evaluateFormula(*arg.left, inputs);
         uint64_t rightValue = evaluateFormula(*arg.right, inputs);
         switch (arg.op) {
         case BinOp::Plus: return leftValue + rightValue;
         case BinOp::Minus: return leftValue - rightValue;
         case BinOp::Mult: return leftValue * rightValue;
         case BinOp::And: return leftValue & rightValue;
         case BinOp::Or: return leftValue | rightValue;
         case BinOp::Xor: return leftValue ^ rightValue;
         default: throw std::runtime_error("Unsupported binary operation");
         }
      }
      }, formula.expr);
}

// Check whether a formula satisfies all examples
bool testFormulaOnExamples(const Formula& formula, const std::vector<Example>& examples) {
   for (const auto& example : examples) {
      uint64_t result = evaluateFormula(formula, example.inputs);
      if (result != example.output) {
         return false;
      }
   }
   return true;
}


int main() {

   auto formulas = generateFormulas(3);
   // printFormulas(formulas);

   const std::string filename = "example_8_and_16bits.json";
   std::vector<Example> examples = readExamples(filename);

   // try {
   //    std::vector<Example> examples = readExamples(filename);

   //    // Display the examples
   //    for (const auto& example : examples) {
   //       std::cout << "Inputs: ";
   //       for (const auto& input : example.inputs) {
   //          std::cout << std::hex << input << " "; // Print inputs in hex
   //       }
   //       std::cout << "Output: " << std::hex << example.output << std::endl; // Print output in hex
   //    }
   // }
   // catch (const std::exception& e) {
   //    std::cerr << "Error: " << e.what() << std::endl;
   // }

   for (const auto& formulaSet : formulas) {
      for (const auto& formula : formulaSet) {
         if (testFormulaOnExamples(*formula, examples)) {
            std::cout << "Found a matching formula:\n";
            printFormula(*formula);
            break;
         }
      }
   }

   return 0;
}