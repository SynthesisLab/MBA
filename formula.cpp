#include <iostream>
#include "formula.h"

using namespace std;

// Variable
Variable::Variable(int v) : var(v) {}

uint32_t Variable::evaluate(const std::vector<uint32_t>& inputs) const {
    return inputs[var];
}

string Variable::toString() const {
    return "Var(" + to_string(var) + ")";
}

// UnaryOperation
UnaryOperation::UnaryOperation(UnOp op, shared_ptr<Formula> operand)
    : op(op), operand(operand) {
}

uint32_t UnaryOperation::evaluate(const vector<uint32_t>& inputs) const {
    uint32_t operandValue = operand->evaluate(inputs);
    switch (op) {
    case UnOp::Not: return ~operandValue;
    default: throw runtime_error("Unsupported unary operation");
    }
}

string UnaryOperation::toString() const {
    switch (op) {
    case UnOp::Not: return "Not(" + operand->toString() + ")";
    default: throw runtime_error("Unsupported unary operation");
    }
}

// BinaryOperation
BinaryOperation::BinaryOperation(BinOp op, shared_ptr<Formula> left, shared_ptr<Formula> right)
    : op(op), left(left), right(right) {
}

uint32_t BinaryOperation::evaluate(const vector<uint32_t>& inputs) const {
    uint32_t leftValue = left->evaluate(inputs);
    uint32_t rightValue = right->evaluate(inputs);
    switch (op) {
    case BinOp::Plus: return leftValue + rightValue;
    case BinOp::Minus: return leftValue - rightValue;
    case BinOp::Mult: return leftValue * rightValue;
    case BinOp::And: return leftValue & rightValue;
    case BinOp::Or: return leftValue | rightValue;
    case BinOp::Xor: return leftValue ^ rightValue;
    default: throw runtime_error("Unsupported binary operation");
    }
}

string BinaryOperation::toString() const {
    switch (op) {
    case BinOp::Plus: return "(" + left->toString() + " + " + right->toString() + ")";
    case BinOp::Minus: return "(" + left->toString() + " - " + right->toString() + ")";
    case BinOp::Mult: return "(" + left->toString() + " * " + right->toString() + ")";
    case BinOp::And: return "(" + left->toString() + " And " + right->toString() + ")";
    case BinOp::Or: return "(" + left->toString() + " Or " + right->toString() + ")";
    case BinOp::Xor: return "(" + left->toString() + " Xor " + right->toString() + ")";
    default: throw runtime_error("Unsupported binary operation");
    }
}

// Formula
Formula::Formula(Variable var) : expr(var) {}
Formula::Formula(BinaryOperation binOp) : expr(binOp) {}
Formula::Formula(UnaryOperation unOp) : expr(unOp) {}

uint32_t Formula::evaluate(const vector<uint32_t>& inputs) const {
    return visit([&inputs](auto&& arg) -> uint32_t {
        return arg.evaluate(inputs);
        }, expr);
}

string Formula::toString() const {
    return visit([](const auto& expr) { return expr.toString(); }, expr);
}

void Formula::printFormula() const {
    cout << toString() << endl;
}