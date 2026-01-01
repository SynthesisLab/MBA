# MBA Expression Synthesis

A high-performance Mixed Boolean-Arithmetic (MBA) expression synthesis tool that generates formulas satisfying given input-output examples. The project provides three implementations with increasing levels of parallelization: sequential, multi-threaded CPU (OpenMP), and GPU (CUDA).

## Overview

This tool performs program synthesis by enumerating and testing formulas of increasing complexity until finding one that satisfies all provided input-output examples. It supports both arithmetic operations (addition, subtraction, multiplication, division, modulo) and bitwise operations (AND, OR, XOR, NOT, shift operations).

## Project Structure

```
.
├── seq/                    # Sequential implementation
│   ├── main.cpp           # Sequential formula generation
│   ├── formula.cpp        # Formula evaluation and representation
│   ├── formula.h          # Formula class definitions
│   ├── read.cpp           # JSON input parsing
│   └── read.h             # Input reading utilities
│
├── parallelcpu/           # Parallel CPU implementation (OpenMP)
│   ├── main.cpp           # Parallel formula generation with OpenMP
│   ├── formula.cpp        # Formula evaluation
│   ├── formula.h          # Formula class definitions
│   ├── read.cpp           # JSON input parsing
│   └── read.h             # Input reading utilities
│
└── gpu/                   # GPU implementation (CUDA)
    ├── mba.cu             # Main CUDA implementation
    ├── mba_alt.cu         # Alternative CUDA implementation
    ├── mba_rpn.cu         # Reverse Polish Notation version
    ├── mbalight.cu        # Lightweight CUDA version
    ├── mbalight_alt.cu    # Alternative lightweight version
    ├── json.hpp           # JSON library (nlohmann/json)
    └── modified_libraries/ # External dependencies
        ├── helpers/       # CUDA helper utilities
        └── warpcore/      # Hash table implementation
```

## Features

### Supported Operations

**Unary Operations:**
- `~` (NOT) - Bitwise NOT
- `-` (NEG) - Negation (GPU only)

**Binary Operations:**
- `+` (Plus) - Addition
- `-` (Minus) - Subtraction
- `*` (Mult) - Multiplication
- `/` (Div) - Division (GPU only)
- `%` (Mod) - Modulo (GPU only)
- `&` (And) - Bitwise AND
- `|` (Or) - Bitwise OR
- `^` (Xor) - Bitwise XOR
- `<<` (LShift) - Left shift (GPU only)
- `>>` (RShift) - Right shift (GPU only)

### Implementation Variants

1. **Sequential (`seq/`)**: Basic enumeration-based synthesis for baseline comparison
2. **Parallel CPU (`parallelcpu/`)**: OpenMP-accelerated parallel exploration
3. **GPU (`gpu/`)**: CUDA-accelerated with hash-based deduplication using warpcore hash sets

## Input Format

The tool reads benchmark files in JSON format containing input-output examples:

```json
{
  "initial": {
    "inputs": {
      "0": {"value": "0x12345678", "size": 32}
    },
    "outputs": {
      "0": {"value": "0xabcdef00", "size": 32}
    }
  },
  "sampling": [
    {
      "inputs": [
        {"value": "0xaaaaaaaa", "size": 32}
      ],
      "outputs": {
        "0": {"value": "0xbbbbbbbb", "size": 32}
      }
    }
  ]
}
```

Place benchmark files in a `benchmarks/` directory (not tracked in git).

## Building

### Sequential Version

```bash
cd seq
g++ -std=c++17 -O3 main.cpp formula.cpp read.cpp -o main
```

### Parallel CPU Version

Requires OpenMP support:

```bash
cd parallelcpu
g++ -std=c++17 -O3 -fopenmp main.cpp formula.cpp read.cpp -o main
```

### GPU Version

Requires NVIDIA CUDA Toolkit (tested with CUDA 11.0+):

```bash
cd gpu
nvcc -std=c++17 -O3 mba.cu -o mba
# Or for alternative versions:
nvcc -std=c++17 -O3 mba_alt.cu -o mba_alt
nvcc -std=c++17 -O3 mba_rpn.cu -o mba_rpn
nvcc -std=c++17 -O3 mbalight.cu -o mbalight
```

## Usage

### Sequential/Parallel CPU

```bash
cd seq  # or parallelcpu
./main
```

The program will:
1. Read examples from `../benchmarks/test.json`
2. Generate and test formulas of increasing size (up to size 10-11)
3. Print the first matching formula
4. Display execution time

### GPU

```bash
cd gpu
./mba
```

The GPU version uses:
- Parallel formula generation across CUDA threads
- Hash-based deduplication to avoid testing duplicate formulas
- Constant memory for output samples (limited to 100 samples)

## Algorithm

The synthesis algorithm follows an enumerative approach:

1. **Initialization**: Start with variable formulas (size 1)
2. **Generation**: For each size n:
   - Apply unary operators to formulas of size n-1
   - Combine formulas of sizes k and n-k-1 with binary operators (1 ≤ k ≤ n-2)
3. **Testing**: Evaluate each generated formula against all examples
4. **Termination**: Stop when a formula matching all examples is found

The GPU version optimizes this by:
- Storing formulas as evaluation results (Characteristic Sets)
- Using hash sets to deduplicate equivalent formulas
- Parallelizing across multiple CUDA blocks/threads

## Performance Considerations

- **Sequential**: Baseline reference, explores formulas systematically
- **Parallel CPU**: Leverages multi-core CPUs with OpenMP parallelization
- **GPU**: Massive parallelization with thousands of threads, optimal for large search spaces

The GPU version includes several optimizations:
- Memory coalescing for efficient global memory access
- Hash-based deduplication to reduce redundant work
- Atomic operations for safe concurrent updates
- Constant memory for frequently accessed data

## Dependencies

- **C++17** compiler (GCC/Clang)
- **OpenMP** (for parallelcpu)
- **NVIDIA CUDA Toolkit** (for gpu, version 11.0+)
- **nlohmann/json** library (included in `json.hpp`)
- **warpcore** hash table library (included in `modified_libraries/warpcore/`)
- **CUDA helpers** (included in `modified_libraries/helpers/`)

## Notes

- Maximum number of samples: 100 (GPU version)
- Formula sizes tested: up to 10-11 nodes
- Measurement mode can be enabled by defining `MEASUREMENT_MODE` in GPU code