#pragma once

#include <stdint.h>

typedef uint32_t(*Op1)(uint32_t);
typedef uint32_t(*Op2)(uint32_t, uint32_t);
typedef uint32_t(*Op3)(uint32_t, uint32_t, uint32_t);

// Device functions for MBA operators
__device__ uint32_t op_and(uint32_t a, uint32_t b) { return a & b; }
__device__ uint32_t op_or(uint32_t a, uint32_t b) { return a | b; }
__device__ uint32_t op_xor(uint32_t a, uint32_t b) { return a ^ b; }
__device__ uint32_t op_add(uint32_t a, uint32_t b) { return a + b; }
__device__ uint32_t op_sub(uint32_t a, uint32_t b) { return a - b; }
__device__ uint32_t op_mul(uint32_t a, uint32_t b) { return a * b; }
__device__ uint32_t op_not(uint32_t a) { return ~a; }

// Number of operators (maximum is 16)
static const int opCount = 7;

// Symbols
static const char* DSLSymbols[] = {
    "&", "|", "^", "+", "-", "*", "~"
};

// Arities (maximum per operator is 3)
static const int DSLArities[] = {
    2, 2, 2, 2, 2, 2, 1
};

// Pointers to device functions
__device__ void* DSLPointers[] = {
    (void*)op_and,
    (void*)op_or,
    (void*)op_xor,
    (void*)op_add,
    (void*)op_sub,
    (void*)op_mul,
    (void*)op_not
};