#ifndef FORMULA_H
#define FORMULA_H

#include <memory>
#include <variant>
#include <vector>

using namespace std;

class Formula;

// Operators
enum class UnOp { Not };
enum class BinOp { Plus, Minus, Mult, And, Or, Xor };

// Variables
class Variable {
public:
    int var;

    explicit Variable(int v);

    uint64_t evaluate(const vector<uint64_t>& inputs) const;

    string toString() const;
};

// Unary operations
class UnaryOperation {
public:
    UnOp op;
    shared_ptr<Formula> operand;

    UnaryOperation(UnOp op, shared_ptr<Formula> operand);

    uint64_t evaluate(const vector<uint64_t>& inputs) const;

    string toString() const;
};

// Binary operations
class BinaryOperation {
public:
    BinOp op;
    shared_ptr<Formula> left;
    shared_ptr<Formula> right;

    BinaryOperation(BinOp op, shared_ptr<Formula> left, shared_ptr<Formula> right);

    uint64_t evaluate(const vector<uint64_t>& inputs) const;

    string toString() const;
};

// Formulas
class Formula {
public:
    variant<Variable, BinaryOperation, UnaryOperation> expr;

    Formula(Variable var);
    Formula(BinaryOperation binOp);
    Formula(UnaryOperation unOp);

    uint64_t evaluate(const vector<uint64_t>& inputs) const;

    string toString() const;
    void printFormula() const;
};

#endif